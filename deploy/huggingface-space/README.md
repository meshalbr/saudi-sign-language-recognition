---
title: Saudi Sign Language Recognition
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Live Saudi Sign Language recognition on CPU, 85.8% on unseen signers
---

# Saudi Sign Language Recognition

Live recognition of 100 Saudi Sign Language signs from your webcam. Your browser finds the
skeleton points with MediaPipe and sends only those points, never the video. A 2 MB
Transformer then classifies the sign on a CPU.

- 85.8% top-1 on signers the model never saw (KArSL-100); the published result is 68.2%
- Sit back so your upper body and both hands are visible, sign, then rest your hands for 1 second
- Open the **Words** tab to see how each sign is made

Code, results and the Docker image: https://github.com/meshalbr/saudi-sign-language-recognition

This Space runs the image that GitHub Actions built and tested
(`ghcr.io/meshalbr/saudi-sign-language-recognition`), in public mode: nothing is stored and
visitors are isolated from each other.
