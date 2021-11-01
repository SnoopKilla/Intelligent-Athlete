# Intelligent-Athlete

## Table of contents
* [Introduction](#introduction "Goto introduction")
* [Technologies](#technologies "Goto technologies")
* [Proof of Concept](#proof-of-concept "Goto proof-of-concept")

## Introduction
The goal of this project is to develop a classifier to keep track of the exercises performed by an athlete during a workout session. In particular, we have considered the following exercises:
* [Push Press](https://www.youtube.com/watch?v=iaBVSJm78ko)
* [Shoulder Press](https://www.youtube.com/watch?v=5yWaNOvgFCM)
* [Push Jerk](https://www.youtube.com/watch?v=V-hKuAfWNUw)
* [Sit-Up](https://www.youtube.com/watch?v=_HDZODOx7Zw)
* [L-Up]()
* [V-Up](https://www.youtube.com/watch?v=7UVgs18Y1P4)

The classifier takes as input the data collected by the accelerometer and gyroscope of [two identical sensors](https://mbientlab.com/store/metamotions-p/) placed on the right wrist and the right ankle of the athlete. The classification pipeline is the following:
1. **Data Pre-Processing**: The data collected by the samples are aligned through a synchronization logic and filtered through a low-pass filter to reduce the noise. After that, gravity and user acceleration are split. Finally, the [sliding window approach](https://www.geeksforgeeks.org/window-sliding-technique/) (which is the core of the following steps) is implemented by splitting the signals into (overlapping) windows and extracting relevant features associated to them.
2. **Binary Classification**: A binary classifier is used to discriminate between periods of activity and periods of rest.
3. **Multiclass Classifier**: Given a period of activity, the multiclass classifier labels such period with the correct exercise label.

![image](https://user-images.githubusercontent.com/80259549/139702050-12497fdf-eb18-4187-bc45-47eac200865e.png)

**I personally contributed to this project by developing the binary classifier (step 2) and implementing the syncronization logic and the sliding window approach (step 1). Moreover, I also developed an Android app as a result visualization tool (see [Proof of Concept](#proof-of-concept "Goto proof-of-concept"))**

## Technologies
Project is created with:
* Python 3.9 (classification logic).
* Android Studio and Chaquopy (Android app).

## Proof of Concept
The following video shows a short workout together with the final result of the classification (displayed by the app).

https://user-images.githubusercontent.com/80259549/139705050-070d1f79-081b-4ee0-8c4d-96abc5b6f1a2.mp4
