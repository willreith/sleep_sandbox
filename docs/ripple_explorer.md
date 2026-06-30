Parameters/decisions:
- passband
- channel (default select from channels with highest power in passband on each shank)
- event duration min/max
- recording (downsampled to 1250Hz)
- bandpass filter method
- plotting duration for envelope in passband
- threshold for significant events (SDs)
- window on either side of detected event to plot

Things to plot:
- once for each set of parameters:
    - ripple band power by depth
    - PSD with ripple band highlighted
    - envelope
    - thresholded envelope
    - thresholded envelope with only events that meet plotting duration criteria
- plots that allow you to sift through detected events

Subsequently added:
- bandpass filter method + parameters
- one plot for events, slide through/jump to next event

Ripple detection method:
The LFP from a selected channel (largest ripple power) was 140 to 250 Hz bandpass filtered by a fourth order Butterworth filter, and then the Hilbert transform were applied to filtered LFP to get ripple band amplitude. Candidate events was detected by choosing the periods that the ripple band amplitude is 2 SD above the mean, peak amplitudes >5 SD, and duration between 30 and 200 ms. After that, SPW-Rs were manually selected from candidate events by looking at the raw LFPs from neighboring channels. (Zhang et al., PNAS, 2021)