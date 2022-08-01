#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:51:52 2022

@author: udaykaranmadaan
#function: import multiple channels of EEG data (at a high sampling frequency)
#then take this data, graph it(in the time domain), compute the FFT and graph the data 
#in the frequeuncy domain then filter the data and graph the filtered data in 
#the frequency domain and compute the iFFT and graph the data in the time domain
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.fftpack import rfft, irfft,ifft,fft
import pandas as pd
from math import log10 , floor


#filtering frequencies 
lowpass=100
highpass=0
notchFreq=60



#colour for each of the graphs
colour1="teal"
colour2="teal"
colour3="teal"
colour4="teal"


#sampling frequency of the data
samplingfreq=250


#import previous eeg data from excel sheet
data = pd.read_excel(r'Book1.xls', sheet_name='Sheet1')


#read sheet for each channel & for the time
#hastag out extra channels when we don't need them

channel_1 = data['first channel'].values
channel_2 = data['second channel'].values
#channel_3 = data['third channel'].values
#channel_4 = data['fourth channel'].values
#channel_5 = data['fifth channel'].values
#channel_6 = data['sixth channel'].values
#channel_7 = data['seventh channel'].values
#channel_8 = data['eighth channel'].values
time = data['time stamp'].values



#defining functions
#-----------
#graph data as a function  of time
def grapherT(time,channel):
    #spacing of ticks on x axis, due to 250Hz data the graph takes a 
    #lot longer if it tries to load every time point 
    #for channel 1 ~4s*250hz=1000 points... this scales and becomes problematic 
    tick_spacing = 1000
    fig, ax = plt.subplots(1,1)
    ax.plot(time,channel,color=colour1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=45)

    #labelling & finally plotting
    plt.xlabel("Time")
    plt.ylabel("uV")
    plt.title("Previous EEG data(cut to 3 mins) channel #")
    plt.show

#graph filtered data as a function of time
def grapherTF(time,channel):

    tick_spacing = 1000
    fig, ax = plt.subplots(1,1)
    ax.plot(time,channel,color=colour2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=45)

    #labelling & finally plotting
    plt.xlabel("Time")
    plt.ylabel("uV")
    plt.title("Previous EEG data(cut to 3 mins) Filtered")
    plt.show

    
#graph filtered data as a function of frequqency
def grapherW(channel,nlen):
   
    
   # see https://www.youtube.com/watch?v=euRpenv4R0A 
   #for learning how what frequencies we need
    f=np.linspace(0,samplingfreq/2,nlen)
    tick_spacing = 15
    fig, ax = plt.subplots(1,1)
    ax.plot(f,abs(channel),color=colour3)
    plt.xlim([-2,150])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=45)

    #labelling & finally plotting
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT Graph")
    
    #manually set the y lim because a data point 
    #skews the rest of the datat from being viewed
    plt.ylim([-100,100000])
    plt.show


#graph filtered data as a function of freq
def grapherWF(channel,nlen):
    
    f=np.linspace(0,samplingfreq/2,nlen)
    tick_spacing = 15
    fig, ax = plt.subplots(1,1)
    ax.plot(f,abs(channel),colour4)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=45)

    #labelling & finally plotting
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim([-2,150])
    plt.ylim([-100,100000])
    plt.title("FFT Graph Filtered")
    plt.show
    
#bandstop filter & notch fitler
def bsnfilter(nlen,channelW,highpass,lowpass,notchFreq):
    
    
    f=np.linspace(0,250/2,nlen)
    
    #bandstop
    
    #high pass filter
    if highpass>=0:
        for i in range(nlen):
            if f[i]<highpass:
                channelW[i]=0
            
            else:
                channelW[i]=channelW[i]
                
    else:
        highpass=highpass
            
            
    #low pass filter
    for i in range(nlen):
        if f[i]>lowpass:
            channelW[i]=0

        else:
            channelW[i]=channelW[i]

    #notch 
    lower = notchFreq-1
    upper = notchFreq+1

    for i in range(nlen):
        if lower <= f[i] <= upper:
            channelW[i]=0
            
        else:
            channelW[i]=channelW[i]
    
    
    return channelW


#rounding function from https://www.delftstack.com/howto/python/round-to-significant-digits-python/#:~:text=digit%20in%20Python.-,Use%20the%20round()%20Function%20to%20Round%20a%20Number%20to,to%20the%20round()%20function.
def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

#determines the % breakdown of brain waves in the signal
def composition(channelw,nlen):
    
    #intializing values
    delta=0
    theta=0
    alpha=0
    beta=0
    gamma=0
    null=0
    
    f=np.linspace(0,samplingfreq/2,nlen)
    
    #add magnitudes
    #definitions for frequency ranges https://www.healthline.com/health-news/your-brain-on-binaural-beats#The-illusion-of-binaural-beats
    for i in range(nlen):
        if f[i]<4 and f[i]>=0:
            delta+=abs(channelw[i])
            
        elif f[i]<8 and f[i]>=4:
            theta+=abs(channelw[i])
            
        elif f[i]<13 and f[i]>=8:
            alpha+=abs(channelw[i])
            
        elif f[i]<30 and f[i]>=13:
            beta+=abs(channelw[i])
            
        elif f[i]>=30:
            gamma+=abs(channelw[i])
            
        else:
            null+=abs(channelw[i]) 
          
    sum=delta+theta+alpha+beta+gamma+null
    
    deltaP=delta/sum*100
    thetaP=theta/sum*100
    alphaP=alpha/sum*100
    betaP=beta/sum*100
    gammaP=gamma/sum*100
    print('Delta Waves Compose ' + str(round_it(deltaP,4)) + '% of Signal\n')
    print('Theta Waves Compose ' + str(round_it(thetaP,4)) + '% of Signal\n')
    print('Alpha Waves Compose ' + str(round_it(alphaP,4)) + '% of Signal\n')
    print('Beta Waves Compose ' + str(round_it(betaP,4)) + '% of Signal\n')
    print('Gamma Waves Compose ' + str(round_it(gammaP,4)) + '% of Signal\n\n')

#just calling all the functions in one function to save time & space
def caller(time,channel,highpass,lowpass,notchFreq):
    
    #graphing the channels
    grapherT(time,channel)
    
    #Fast fourier transform
    channel_w=rfft(channel)
    
    #length of channel
    nlen=len(channel)
    
    #graphing the data in the frequency domain
    grapherW(channel_w,nlen)
    
    #filtering & graphing the filtered data in the frequency domain
    channel_wf=bsnfilter(nlen,channel_w,highpass,lowpass,notchFreq)
    grapherWF(channel_wf,nlen)
    
    #inverse fast fourier transform of the filtered data
    channel_tf=irfft(channel_wf)
    
    #graph the filtered data in the time domain
    grapherTF(time,channel_tf)

    #finally calculate composition
    composition(channel_wf,nlen)


#caller(time,channel_1,highpass,lowpass,notchFreq)
caller(time,channel_2,highpass,lowpass,notchFreq)
