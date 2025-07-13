# HDSwift

## Overview  
HDSwift is a project aimed at leveraging the Apple Neural Engine (ANE) to optimize a machine learning algorithm originally provided by my professor. The goal was to explore how to best utilize the ANE alongside CPU and GPU compute to improve efficiency.

## Purpose  
This project investigates the use of ANE for accelerating machine learning operations. We experimented with CPU, GPU, and Neural Processing Unit (NPU) computations to understand the benefits and challenges of integrating Apple’s hardware acceleration into ML workflows.

## Technologies Used  
- CoreMLTools  
- DispatchQueue (for concurrency management)

## Setup and Usage  
Currently, the project is somewhat hardcoded and lacks a flexible setup. You can run the project via the main “Run” button in Xcode. Further refinements to installation and configuration may be needed.

## Key Features  
- Utilizes Apple Neural Engine for machine learning operations.  
- Explores the performance differences between CPU, GPU, and NPU compute.  
- Demonstrates challenges with model instantiation when using ANE for certain operations.

## Limitations  
- The current implementation instantiates a new model on every multiplication, which reduces the overall effectiveness of the ANE optimization.

## Contributions  
Contributions and issue reporting are not currently supported.

## License  
License information is not specified.
