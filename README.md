# Inverse-Scattering-Problem

Code for simulating the inverse scattering problem in the context of indoor imaging using Wi-Fi waves. The goal of the inverse scattering problem as useful in indoor imaging is to reconstruct the location, shape and refractive index of an object kept in an indoor region using wireless signals. The wireless signals are transmitted and received using Wi-Fi sensors operating at 2.4 GHz frequency.

### Problem Setup
 
### Solving the forward problem
The forward problem involves estimating the RSSI values measured by each sensor based on objects placed in the region to be imaged. This is done by using the Method of Moments approach detailed in [1]. This constitutes the measurement data that is then used to solve the inverse problem.

### Solving the inverse problem
The inverse problem is to reconstruct the image of the scattering object (including its shape, location and refractive index) using the measurements obtained through the forward problem. It can be formulated as a problem of minimizing the sum of squared errors $||y - Ax||$ where $y$ is the measurement vector, $A$ is the inverse model matrix and $x$ is the reconstruction of the imaging region. In our particular use case of inverse scattering, this problem is highly ill-posed (i.e. number of unknowns are much more than the number of measurements), which is why we use regularization to solve the problem. We have implemented theh Tikhonov regularization for this purpose.

