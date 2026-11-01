# DreamWeaver AI — Story Wall

A fully local, offline, **voice-driven storytelling companion** for a young child.
A child speaks into a **USB microphone**; a local language model decides whether he
wants to talk or wants a picture; a text-to-image model paints the scene; an
**HDMI projector** displays it on the wall and plays the character's spoken reply
back through its own built-in speakers — over the same HDMI cable.

No cloud calls. No accounts. No telemetry. Everything runs on one machine.

> **⚠ About this repository — content policy**
> This repository is an **architecture and design documentation only**.
> Due to copyright concerns — the third-party model weights (each under its own
> licence and not redistributable here), GPL-3.0 components in the image pipeline,
> and private family content in recorded audio and generated images — **no source
> code, prompts, persona text, configuration files, or generated content are
> published in this repository.** Only the system design, the measured performance
> data, and the reasoning behind each decision are shared.

---

## 1. The hardware foundation: AMD "Strix Halo" with 128 GB unified memory

The entire design is only possible because of one platform choice: an
**AMD Ryzen AI Max+ 395 ("Strix Halo")** mini PC — 16C/32T Zen 5 CPU with the
**Radeon 8060S iGPU (gfx1151, RDNA 3.5)** and **128 GB LPDDR5X unified memory**.

Why this matters:

| Property | Value | Consequence |
|---|---|---|
| Usable unified memory | ~122 GiB | Four AI models stay **resident in GPU-addressable memory at the same time** |
| GTT aperture (GPU-visible RAM) | ~123 GiB | There is **no VRAM wall** — the GPU addresses system RAM directly |
| Measured RAM bandwidth | ~215 GB/s | This, not capacity, is the real speed ceiling (vs ~2.4 TB/s on desktop HBM) |
| Fixed VRAM carve-out | 0.5 GiB (none needed) | The iGPU simply reads/writes the shared pool |

A discrete 24 GB graphics card **cannot run this design at all** — four models
(~49 GiB combined) would not fit. This is a hard architectural limit, not a
tuning problem. Conversely, bandwidth is what makes image generation take ~8.5 s
rather than ~1 s, which in turn shapes the interaction design (§6).

Host stack: Ubuntu LTS, ROCm 7.1 userspace. The LLM and speech-to-text run on the
**Vulkan (RADV)** stack; the TTS and image pipelines run in **ROCm** containers.

Peripherals:

| Device | Role |
|---|---|
| USB microphone | Child's voice input; its physical mute-style button doubles as a push-to-talk trigger (exposed as an input event) |
| HDMI projector | The "wall": video **and** audio both travel over one HDMI cable — the projector's built-in speakers are the character's voice |
| Console TTY switch | The projector takes a dedicated virtual terminal (tty3) with its own compositor, so the main desktop is never disturbed |

---

## 2. System topology

```
                 USB microphone (voice in + physical button)
                                  |
                  +---------------+----------------+
                  |                                |
            wake-word listener              push-to-talk variant
         (continuous capture, VAD)          (button press -> turn)
                  |
        speech segments -> ASR node (loopback :8055)
                  |          Whisper large-v3-turbo, Vulkan
             transcript
                  |
                  v
              THE ROUTER (turn)  -------------------------------+
                  |                |                           |
             LLM node :8090   TTS node :8056              image job
         Qwen3.6-35B-A3B    Qwen3-TTS 1.7B                  |
         GGUF Q4_K_XL       bf16, ROCm ctr                  v
                  |                |                   ComfyUI :8188 (container)
                  |                |                   Z-Image-Turbo, bf16
                  v                v                          |
              local audio playback  <--- sentences stream in as generated
                  |                                           |
                  v                                           v
            +----------------------  WALL  ----------------------+
            |  compositor + media player on tty3, local IPC     |
            |  HDMI  ---> projector  ---> wall image + speakers |
            +----------------------------------------------------+
```

Every internal link is loopback HTTP or a Unix socket on the machine itself.
**Nothing binds a routable address** — the security posture is enforced by
binding, not trust, since none of the inference servers have authentication.

---

## 3. The model runtime: which models, what quant, how much memory

All figures below were **measured on the reference machine**, not from spec sheets.
Total model footprint on disk: ~51 GB.

| Stage | Model | Precision / quantization | Served by | Disk | GPU-resident memory | Measured latency |
|---|---|---|---|---|---|---|
| **Speech → text** | Whisper large-v3-turbo | GGML FP16 | whisper-style server, Vulkan | 1.6 GB | ~2 GiB | **0.23 s** per 6 s clip |
| **Intent + language (LLM)** | Qwen3.6-35B-A3B (MoE, ~3B active) | GGUF **UD-Q4_K_XL** (4-bit K-quant, unlimited-decay mix) | llama.cpp server, Vulkan, mmap disabled | 23 GB | **22.7 GiB** (weights + KV cache) | prefill 1594 tok/s @512 tok; **61.65 tok/s** decode |
| **Text → speech** | Qwen3-TTS 1.7B (voice-design) | **BF16** | Python HTTP server in ROCm container | 4.3 GB | ~4 GiB | RTF **0.86–0.94** warm; 11 s model load |
| **Text → image** | Z-Image-Turbo (Lumina2 family) + mandatory distilled-step LoRA | **BF16** throughout (UNet ~11.5 GiB + Qwen3-4B text encoder ~7.5 GiB), CFG locked at 1.0, 8 steps | ComfyUI in ROCm container | 20 GB | **~19 GiB** | **8.5 s** at 1024×576; 13.3 s at 1280×720; ~24 s cold first image |

Quantization notes:

- **Q4_K_XL** was chosen for the LLM as the quality/size sweet spot for a 35B MoE;
  because it is served with memory mapping disabled (a gfx1151-specific driver
  pathology above 64 GB), the whole 22.7 GiB is pinned resident in the GTT
  aperture for the entire session.
- The image model's distillation LoRA is what allows **8 sampling steps**;
  without it the model wants 20–50 steps and the interaction budget collapses.
  Classifier-free guidance is fixed at 1.0 because the turbo variant is
  guidance-distilled — anything higher silently doubles latency.
- The TTS model runs in BF16 on ROCm; a per-shape warm-up is performed at load so
  no child ever waits on the first-call solver search.

### Total memory budget (everything resident, wall up)

| | |
|---|---|
| GPU-addressed (GTT) in use | **49.1 GiB** of ~123 GiB |
| Host RSS of the four GPU processes | 22.8 GiB |
| System RAM used / available | 74 GiB / 48 GiB |

Rough split of the 49.1 GiB: LLM 22.7 + image pipeline ~19 + TTS ~4 + ASR ~2,
remainder in buffers and the compositor. `story stop` releases it all —
**nothing runs at boot**; the machine is an ordinary computer until story mode is
deliberately started.

---

## 4. One turn, in measured order

Speech in → answer out on the warm system:

| t | what happens |
|---|---|
| 0.00 s | capture ends (energy-gated), audio handed to ASR |
| 0.23 s | transcript available |
| ~1.1 s | first LLM token |
| ~1.6 s | the router already knows the *intent* branch (it parses the streaming JSON incrementally, not at the end) |
| talk branch | sentences are pushed to the speech queue as they complete; speaking starts ~2 s |
| draw branch | a "painting…" overlay goes up on the projector; rendering begins |
| 10.1 s | image appears on the wall |
| 10.4 s | first spoken sentence of the character's comment plays over HDMI audio |
| ~17 s | turn complete |

The ordering is deliberate: for a *draw* request the picture is composited
**first** and only then described — a description of a picture that appears three
seconds later reads as a lie to a child.

---

## 5. The intent contract (design, not code)

Each turn, the LLM must return exactly one small JSON object (schema-enforced
server-side, strict mode) with three fields:

1. **`intent`** — an enum: `talk` | `draw` | `redraw`. Typed as an enum on purpose:
   the router needs a branch, not a suggestion.
2. **`voice_line`** — what the character says out loud, Mandarin, ≤30 words
   (Mandarin because the TTS voice design is Mandarin; English input emerges as
   unread romanisation).
3. **`image_prompt`** — English, only for `draw`/`redraw` (the image model's text
   encoder performs far better in English; mixing the two language choices is the
   easiest way to break the system silently).

If the child's utterance was itself a drawing request, a flag forces the draw
branch regardless of the model's classification — a hedging model can never
swallow the child's request.

---

## 6. Design rules that came from measurement

1. **She only hears the room while idle.** The listener is blocked for the whole
   turn — the mic is physically deaf while her own voice comes out of the
   projector speakers in front of it (self-hearing otherwise loops forever).
2. **Never let the ASR see silence.** It invents fluent sentences from nothing, so
   every capture path has an energy gate first.
3. **The character never draws on her own initiative.** Naming an animal or
   describing the day must come back as *talk*, inviting the child to draw it
   himself — a toy that paints everything becomes a television.
4. **Nothing persists about previous pictures.** Each request is answered on its
   own terms; no image ever feeds back into the models.
5. **The fast path is resident; anything slow is a queue that borrows the GPU.**
   Long jobs (e.g. video) are a separate system sharing only the GPU, never a
   stage inside a turn.

---

## 7. Display and audio output path

- Story mode moves the console to a dedicated virtual terminal and runs a
  lightweight **compositor plus a media player** there, owned by root-started
  system services (group membership is resolved at exec time, which is what
  grants the push-to-talk button access without autologin).
- The router talks to the media player through a **Unix-socket IPC**: show image,
  play video, overlay progress text, clear.
- **One HDMI cable carries both the picture and the character's voice** to the
  projector's built-in speakers — no separate audio wiring, and the audio output
  device is the projector's HDMI endpoint.
- No autologin kiosk: the machine stays a normal desktop; the wall is explicitly
  started and torn down.

---

## 8. Measured end-to-end summary

| Metric | Value |
|---|---|
| Full turn: speech → picture → voice | **~17 s** |
| LLM decode (35B MoE @ Q4_K_XL) | 61.65 tok/s |
| Image generation (1024×576, 8 steps) | 8.5 s |
| Speech synthesis real-time factor | 0.86–0.94 |
| GPU memory all-in | ~49 GiB of ~123 GiB aperture |
| Models resident simultaneously | 4 |

## 9. Licence and redistribution

This repository contains **only documentation**. No implementation code, prompts,
persona text, model weights, audio, or generated images are distributed here,
due to copyright and licence concerns: the language, speech, and image models are
each bound by their own upstream licences and must be downloaded separately from
their official sources, and components of the image pipeline are GPL-3.0. If you
are rebuilding a system like this, read the licence terms of every model you
download before re-hosting or redistributing anything.
