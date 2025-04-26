# Core Functionality
The first part of the plan is designing the core functionality of the library. The core functionality consist of libraries that perform fundamental computations and mathematical formulas, such as free space path loss, noise conversions, and antenna gain calculations.

It is critical for these libraries to be well tested and free of any corner or edge cases. Since this code will be used for planning deep space missions as well as LEO missions, extensive testing at frequencies up to Ka band, and distances as far as 2 million km or more are required. Additionally, signal levels are significantly lower and bit rates are often lower for these deep space missions, so testing code at rates as low as 100 bps, and signal levels as low as -180 dBW is also required.

# Components
The second part of the plan is to design the components of the library. These components are the building blocks of the library, and are used to create complex systems. The components are designed to be as generic as possible, so that they can be used in a variety of different systems.

These components abstract standard RF/microwave component behaviors, such as amplifiers, filters, and attenuators. They also abstract more complex components, such as receivers and transmitters. However, these components do not model phase related behavior. All losses are computed based on magnitude only, so there is some inherent inaccuracy due to this limitation. 

If phase accurate modeling is required, we recommend using scikit-rf for those networks

