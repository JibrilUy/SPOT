# Spatial Positioning through Optical Tracking for the Bulacan State University E-library

An undergraduate thesis project focused on automated crowd monitoring and occupancy tracking using computer vision.

## Developed and Authored by:
* Budaño, Gabriel C. 
* Lacuna, Andrew Gabriel E.
* Panganiban, Jayve Jian J. 
* Cortez, Rexine Bradley d.L.

## Overview
**SPOT** (Spatial Positioning through Optical Tracking) is an automated system developed as an undergraduate thesis for the Bulacan State University E-library. The system utilizes CCTV camera feeds and deep learning object detection architectures (such as YOLO) alongside tracking algorithms to detect, count, and monitor individuals within the facility in real time.

## Key Features
* Real-time person detection and tracking using optical CCTV feeds.
* PyQt6-based desktop application featuring modular UI components, custom styling, and navigation controllers.
* Spatial positioning and analytical data handling for library occupancy management.
* Modular integration supporting Firebase NoSQL databases and local RTSP/WebRTC video streaming workflows.

## Technology Stack
* **Language:** Python
* **GUI Framework:** PyQt6
* **Computer Vision & Tracking:** OpenCV, YOLO, filtering algorithms
* **Backend & Hosting:** PostGres SQL

## Project Structure
```text
SPOT/
├── assets/
│   └── resource fles/
├── controllers/
│   └── Connects Front and Backend/
├── core/
│   └── Camera Processing/
├── gui/
│   └── PyQT generated files/
├── main.py
└── README.md
