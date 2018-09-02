# regression_selfi
The program requires dlib library and csv-text header only library
It is doing a regression (least mean square SVM) also known K rigid regression available in dlib library. It takes input from csv file. that containts the following
Image : X : Y : Z : Zoom : Pitch : Yaw:::SLS : SW/BH : SW/IW : FH/BH : FH/SW : 
Where SLS = should length slope
SW/BH = soulder width/body height
SW/IW = shoulder width normalized with Image width
FH/BH = face height/body height

It train the KRR regression function using SLS : SW/BH : SW/IW : FH/BH : FH/SW  as sample input feature space and outputs 
Image : X : Y : Z : Zoom : Pitch. As this is scaler regression, therefore we need seperate decision function for each output.

Work is support by NTUST Intelligent Robotics Lab NTUST

Team members: Waqar Shahid Qureshi, Manshoor Ali, Prof. Jerry Lin


