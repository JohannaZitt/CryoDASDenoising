"""

Different possibilties to calculate SNR values:



Definition of SNR:
 - ratio of signal power to noise power. A ratio higher than 1:1 (greater than 0 dB) indicates more signal than noise. (wikipedia)
 - the energy (or sometimes amplitude) of the signal divided by the remaining (noise) energy at that time. SNR is
   hard ro determine because of the difficulty in separating out the signal. (https://wiki.seg.org/wiki/Dictionary:Signal-to-noise_ratio_(S/N))


1. naive approach: (https://www.researchgate.net/post/How-can-I-evaluate-the-signal-noise-ratios-of-seismic-signals-using-Matlab)
        - since the signal should have mean zero, the variance is computed instead.
        - S_N + S_X = S_N+X holds true, when the mean is zero for signal and noise.
        - estimate the variance S_N of the noise
        - estimate the variance S_N+X of the seismic signal corrupted with noise
        - SNR = S_X/S_N = (S_N+X - S_N) / S_N
2. naive approach: (chatGPT)
        - The SNR is typically defined as the ratio of the signal's power (in a relevant time window) to the noise's power (in another window).
        - the signal's power is calculated as the root mean square value of the signal
3. naive approach (van den Ende)
        - van den Ende is not calculating the SNR value od DAS records
        - in van den Ende's paper the linear SNR value is depicted (also in mine)

Plan: calculating power or energy or variance over a short window localized by the maximum amplitude and divide it
      by a long window outside of the short window. Just cut the short window out of the long window. Using parameters
      as for STA/LTA?
      Dealing with the DAS data: computing SNR value for every channel separately and building mean over all values
      Using linear SNR!!
      Taking median over all icequakes

Different possibilities to calculate the snr:
- RMS (root mean square): sqr(1/n(x_1² + ... + x_n²))
- variance: 1/n((x_1 - x_mean) + ... + (x_n - x_mean))
- energy: 1/N(|x_1|² + ... + |x_n|²) = power
-









"""