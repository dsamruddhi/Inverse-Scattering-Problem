# Inverse Scattering Problem

Code for simulating the inverse scattering problem in the context of indoor imaging using Wi-Fi waves. The goal of the inverse scattering problem as useful in indoor imaging is to reconstruct the location, shape and refractive index of an object kept in an indoor region using wireless signals. The wireless signals are transmitted and received using Wi-Fi sensors operating at 2.4 GHz frequency.

### Problem Setup
The figure below shows an example of an inverse scattering problem setup containing Wi-Fi transmitters and receivers denoted as Tx and Rx respectively. 

<img align="left" src="https://user-images.githubusercontent.com/5306916/138258361-37900821-9850-43a4-8d29-7bb9d308dc5b.png" width="400" height="300">

Each transmitted receiver link makes up one measurement. Thus, if `t` is the number of trasmitters and `r` is the number of receivers, the total number of links would be `t*r`.
Thus, we have a total of `t*r` measurements.

The area to be imaged is divided into `200x200` grids for the forward problem and `50x50` for the inverse problem.


### Solving the forward problem
The forward problem involves estimating the RSSI values measured by each sensor based on objects placed in the region to be imaged. This is done by using the Method of Moments approach detailed in [1]. This constitutes the measurement data that is then used to solve the inverse problem.

### Solving the inverse problem
The inverse problem is to reconstruct the image of the scattering object (including its shape, location and refractive index) using the measurements obtained through the forward problem. It can be formulated as a problem of minimizing the sum of squared errors <img src="https://render.githubusercontent.com/render/math?math=||y - Ax||^2"> where `y` is the measurement vector, `A` is the inverse model matrix and `x` is the reconstruction of the imaging region. In our particular use case of inverse scattering, this problem is highly ill-posed (i.e. number of unknowns are much more than the number of measurements), which is why we use regularization to solve the problem. We have implemented the Tikhonov regularization for this purpose, which changes the optimization problem to minimizing <img src="https://render.githubusercontent.com/render/math?math=||y - Ax||^2%20%2B%20\lambda%20||\Theta x||">
 where <img src="https://render.githubusercontent.com/render/math?math=\Theta"> is the Tikhonov operator.

### References

[1] Chen, Xudong. Computational methods for electromagnetic inverse scattering. John Wiley & Sons, 2018.

[2] Dubey, Amartansh, et al. "An Enhanced Approach to Imaging the Indoor Environment Using WiFi RSSI Measurements." IEEE Transactions on Vehicular Technology 70.9 (2021): 8415-8430.
