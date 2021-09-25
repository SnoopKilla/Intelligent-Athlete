import binary
import multiclass

wA = "TestSet/wristAcc.csv"
wG = "TestSet/wristGyr.csv"
aA = "TestSet/ankleAcc.csv"
aG = "TestSet/ankleGyr.csv"
result = multiclass.classifier(binary.classifier(wA,wG,aA,aG))
print(result)
