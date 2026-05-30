# Advanced: Image Acquisition

!!! info "Going deeper"
    The content below is **supplementary material**. It provides a roadmap for exploring industrial image acquisition beyond the dataset-based simulations used in this project.

## Why Acquisition Matters

In production machine vision systems, the camera and acquisition pipeline are often the weakest link. The best ML model in the world produces garbage if the input images are:

- Blurry (wrong exposure time for conveyor speed)
- Inconsistent (lighting changes, triggering jitter)
- Missing (dropped frames due to bandwidth saturation)
- Out of sync (multi-camera setups with timing drift)

Mastering acquisition is what separates a lab demo from a deployed system.

## Industrial Camera Standards

### GigE Vision
The most common interface for industrial cameras. Uses standard Ethernet infrastructure with a defined protocol for camera discovery, configuration, and image streaming. Bandwidth: ~1 Gbps (125 MB/s), extendable to 10 GigE.

### Camera Link
High-bandwidth point-to-point interface using dedicated frame grabber cards. Up to 6.8 Gbps. Used when GigE bandwidth isn't sufficient (high-resolution line-scan at speed).

### USB3 Vision
USB 3.0 based standard. Simpler setup than GigE, good bandwidth (~5 Gbps), but limited cable length (~5m). Common in lab and benchtop systems.

### CoaXPress (CXP)
Coaxial cable interface. Up to 12.5 Gbps per link, supports long cable runs. Used in high-speed, high-resolution applications.

## GenICam and GenTL

**GenICam** is the standard that unifies camera configuration across all interface types. Instead of writing vendor-specific code, you use a generic API that reads the camera's XML feature description and exposes settings like exposure time, gain, trigger mode, etc.

**GenTL** (GenICam Transport Layer) is the acquisition layer. It defines how image data flows from camera to application through a standardized producer-consumer model:

```
Camera → GenTL Producer → GenTL Consumer (your application)
```

### Harvesters (Python)

[Harvesters](https://github.com/genicam/harvesters) is a Python library that acts as a GenTL consumer. It can talk to any camera through any GenTL producer.

```python
from harvesters.core import Harvester

h = Harvester()
h.add_file("/path/to/gentl_producer.cti")  # Vendor-specific producer
h.update()

# Connect to first camera found
ia = h.create()
ia.start()

# Capture frames
with ia.fetch() as buffer:
    image = buffer.payload.components[0].data
    # Process image...

ia.stop()
ia.destroy()
h.reset()
```

### Where to Get GenTL Producers

- **Basler**: Ships with Pylon SDK — download from [baslerweb.com](https://www.baslerweb.com/en/downloads/software-downloads/)
- **FLIR/Teledyne**: Ships with Spinnaker SDK
- **Allied Vision**: Ships with Vimba X SDK
- **Aravis** (Linux only): Open-source GenTL producer for GigE cameras — [github.com/AravisProject/aravis](https://github.com/AravisProject/aravis)

## Key Concepts to Explore

### Trigger Modes
- **Free-running**: Camera captures continuously at a set frame rate
- **Software trigger**: Application signals when to capture
- **Hardware trigger**: External signal (encoder pulse, PLC output) initiates capture
- **Action commands**: GigE Vision feature for synchronized multi-camera capture

### Buffer Management
Industrial cameras stream at rates that can exceed processing speed. Buffer strategies include:

- **Circular buffer**: Fixed-size ring buffer, oldest frames overwritten if consumer is slow
- **Queue buffer**: Frames queued until consumed, backpressure if full
- **Newest-only**: Only the most recent frame is available, all older frames discarded

### Line-Scan Acquisition
Instead of capturing 2D frames, line-scan cameras capture one line of pixels per trigger. The 2D image is built up line-by-line as the object moves past the camera on a conveyor. Key parameters:

- **Line rate**: Lines per second (typically 10K–200K)
- **Encoder resolution**: Trigger pulses per mm of conveyor travel
- **Image height**: How many lines to accumulate before sending a complete image

## Recommended Resources

- [Basler Vision Campus](https://www.baslerweb.com/en/vision-campus/) — Free tutorials on industrial vision fundamentals
- [EMVA GenICam Standard](https://www.emva.org/standards-technology/genicam/) — Official standard documents
- [Edmund Optics Knowledge Center](https://www.edmundoptics.com/knowledge-center/) — Optics and imaging guides
- [A3 (Association for Advancing Automation)](https://www.automate.org/) — Industry standards and best practices
- [Harvesters Documentation](https://harvesters.readthedocs.io/) — Python GenTL consumer library
