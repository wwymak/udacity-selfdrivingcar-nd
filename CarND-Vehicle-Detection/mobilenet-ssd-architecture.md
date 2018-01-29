### Mobilenet SSD architecture

Everything from `conv1` to `conv_pw_13_relu` is from the official keras implementation of MobileNet
`input_mean_norm` and `input_scaler` are normalisation layers. Classes 4 to 7 are the classification
layers of the SSD, box4 to 7 (together with the AnchorBoxes layers) are for predicting bounding boxes, and the  
concatenation layers are to put the various class/boxes features together for the final prediction output.

The AnchorBoxes are precomputed guesses of where the boxes should be-- the network then learns to adjust them according
to the training data.

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 300, 480, 3)  0
__________________________________________________________________________________________________
input_mean_norm (Lambda)        (None, 300, 480, 3)  0           input_1[0][0]
__________________________________________________________________________________________________
input_scaler (Lambda)           (None, 300, 480, 3)  0           input_mean_norm[0][0]
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 150, 240, 32) 864         input_scaler[0][0]
__________________________________________________________________________________________________
conv1_bn (BatchNormalization)   (None, 150, 240, 32) 128         conv1[0][0]
__________________________________________________________________________________________________
conv1_relu (Activation)         (None, 150, 240, 32) 0           conv1_bn[0][0]
__________________________________________________________________________________________________
conv_dw_1 (DepthwiseConv2D)     (None, 150, 240, 32) 288         conv1_relu[0][0]
__________________________________________________________________________________________________
conv_dw_1_bn (BatchNormalizatio (None, 150, 240, 32) 128         conv_dw_1[0][0]
__________________________________________________________________________________________________
conv_dw_1_relu (Activation)     (None, 150, 240, 32) 0           conv_dw_1_bn[0][0]
__________________________________________________________________________________________________
conv_pw_1 (Conv2D)              (None, 150, 240, 64) 2048        conv_dw_1_relu[0][0]
__________________________________________________________________________________________________
conv_pw_1_bn (BatchNormalizatio (None, 150, 240, 64) 256         conv_pw_1[0][0]
__________________________________________________________________________________________________
conv_pw_1_relu (Activation)     (None, 150, 240, 64) 0           conv_pw_1_bn[0][0]
__________________________________________________________________________________________________
conv_dw_2 (DepthwiseConv2D)     (None, 75, 120, 64)  576         conv_pw_1_relu[0][0]
__________________________________________________________________________________________________
conv_dw_2_bn (BatchNormalizatio (None, 75, 120, 64)  256         conv_dw_2[0][0]
__________________________________________________________________________________________________
conv_dw_2_relu (Activation)     (None, 75, 120, 64)  0           conv_dw_2_bn[0][0]
__________________________________________________________________________________________________
conv_pw_2 (Conv2D)              (None, 75, 120, 128) 8192        conv_dw_2_relu[0][0]
__________________________________________________________________________________________________
conv_pw_2_bn (BatchNormalizatio (None, 75, 120, 128) 512         conv_pw_2[0][0]
__________________________________________________________________________________________________
conv_pw_2_relu (Activation)     (None, 75, 120, 128) 0           conv_pw_2_bn[0][0]
__________________________________________________________________________________________________
conv_dw_3 (DepthwiseConv2D)     (None, 75, 120, 128) 1152        conv_pw_2_relu[0][0]
__________________________________________________________________________________________________
conv_dw_3_bn (BatchNormalizatio (None, 75, 120, 128) 512         conv_dw_3[0][0]
__________________________________________________________________________________________________
conv_dw_3_relu (Activation)     (None, 75, 120, 128) 0           conv_dw_3_bn[0][0]
__________________________________________________________________________________________________
conv_pw_3 (Conv2D)              (None, 75, 120, 128) 16384       conv_dw_3_relu[0][0]
__________________________________________________________________________________________________
conv_pw_3_bn (BatchNormalizatio (None, 75, 120, 128) 512         conv_pw_3[0][0]
__________________________________________________________________________________________________
conv_pw_3_relu (Activation)     (None, 75, 120, 128) 0           conv_pw_3_bn[0][0]
__________________________________________________________________________________________________
conv_dw_4 (DepthwiseConv2D)     (None, 38, 60, 128)  1152        conv_pw_3_relu[0][0]
__________________________________________________________________________________________________
conv_dw_4_bn (BatchNormalizatio (None, 38, 60, 128)  512         conv_dw_4[0][0]
__________________________________________________________________________________________________
conv_dw_4_relu (Activation)     (None, 38, 60, 128)  0           conv_dw_4_bn[0][0]
__________________________________________________________________________________________________
conv_pw_4 (Conv2D)              (None, 38, 60, 256)  32768       conv_dw_4_relu[0][0]
__________________________________________________________________________________________________
conv_pw_4_bn (BatchNormalizatio (None, 38, 60, 256)  1024        conv_pw_4[0][0]
__________________________________________________________________________________________________
conv_pw_4_relu (Activation)     (None, 38, 60, 256)  0           conv_pw_4_bn[0][0]
__________________________________________________________________________________________________
conv_dw_5 (DepthwiseConv2D)     (None, 38, 60, 256)  2304        conv_pw_4_relu[0][0]
__________________________________________________________________________________________________
conv_dw_5_bn (BatchNormalizatio (None, 38, 60, 256)  1024        conv_dw_5[0][0]
__________________________________________________________________________________________________
conv_dw_5_relu (Activation)     (None, 38, 60, 256)  0           conv_dw_5_bn[0][0]
__________________________________________________________________________________________________
conv_pw_5 (Conv2D)              (None, 38, 60, 256)  65536       conv_dw_5_relu[0][0]
__________________________________________________________________________________________________
conv_pw_5_bn (BatchNormalizatio (None, 38, 60, 256)  1024        conv_pw_5[0][0]
__________________________________________________________________________________________________
conv_pw_5_relu (Activation)     (None, 38, 60, 256)  0           conv_pw_5_bn[0][0]
__________________________________________________________________________________________________
conv_dw_6 (DepthwiseConv2D)     (None, 19, 30, 256)  2304        conv_pw_5_relu[0][0]
__________________________________________________________________________________________________
conv_dw_6_bn (BatchNormalizatio (None, 19, 30, 256)  1024        conv_dw_6[0][0]
__________________________________________________________________________________________________
conv_dw_6_relu (Activation)     (None, 19, 30, 256)  0           conv_dw_6_bn[0][0]
__________________________________________________________________________________________________
conv_pw_6 (Conv2D)              (None, 19, 30, 512)  131072      conv_dw_6_relu[0][0]
__________________________________________________________________________________________________
conv_pw_6_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_pw_6[0][0]
__________________________________________________________________________________________________
conv_pw_6_relu (Activation)     (None, 19, 30, 512)  0           conv_pw_6_bn[0][0]
__________________________________________________________________________________________________
conv_dw_7 (DepthwiseConv2D)     (None, 19, 30, 512)  4608        conv_pw_6_relu[0][0]
__________________________________________________________________________________________________
conv_dw_7_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_dw_7[0][0]
__________________________________________________________________________________________________
conv_dw_7_relu (Activation)     (None, 19, 30, 512)  0           conv_dw_7_bn[0][0]
__________________________________________________________________________________________________
conv_pw_7 (Conv2D)              (None, 19, 30, 512)  262144      conv_dw_7_relu[0][0]
__________________________________________________________________________________________________
conv_pw_7_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_pw_7[0][0]
__________________________________________________________________________________________________
conv_pw_7_relu (Activation)     (None, 19, 30, 512)  0           conv_pw_7_bn[0][0]
__________________________________________________________________________________________________
conv_dw_8 (DepthwiseConv2D)     (None, 19, 30, 512)  4608        conv_pw_7_relu[0][0]
__________________________________________________________________________________________________
conv_dw_8_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_dw_8[0][0]
__________________________________________________________________________________________________
conv_dw_8_relu (Activation)     (None, 19, 30, 512)  0           conv_dw_8_bn[0][0]
__________________________________________________________________________________________________
conv_pw_8 (Conv2D)              (None, 19, 30, 512)  262144      conv_dw_8_relu[0][0]
__________________________________________________________________________________________________
conv_pw_8_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_pw_8[0][0]
__________________________________________________________________________________________________
conv_pw_8_relu (Activation)     (None, 19, 30, 512)  0           conv_pw_8_bn[0][0]
__________________________________________________________________________________________________
conv_dw_9 (DepthwiseConv2D)     (None, 19, 30, 512)  4608        conv_pw_8_relu[0][0]
__________________________________________________________________________________________________
conv_dw_9_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_dw_9[0][0]
__________________________________________________________________________________________________
conv_dw_9_relu (Activation)     (None, 19, 30, 512)  0           conv_dw_9_bn[0][0]
__________________________________________________________________________________________________
conv_pw_9 (Conv2D)              (None, 19, 30, 512)  262144      conv_dw_9_relu[0][0]
__________________________________________________________________________________________________
conv_pw_9_bn (BatchNormalizatio (None, 19, 30, 512)  2048        conv_pw_9[0][0]
__________________________________________________________________________________________________
conv_pw_9_relu (Activation)     (None, 19, 30, 512)  0           conv_pw_9_bn[0][0]
__________________________________________________________________________________________________
conv_dw_10 (DepthwiseConv2D)    (None, 19, 30, 512)  4608        conv_pw_9_relu[0][0]
__________________________________________________________________________________________________
conv_dw_10_bn (BatchNormalizati (None, 19, 30, 512)  2048        conv_dw_10[0][0]
__________________________________________________________________________________________________
conv_dw_10_relu (Activation)    (None, 19, 30, 512)  0           conv_dw_10_bn[0][0]
__________________________________________________________________________________________________
conv_pw_10 (Conv2D)             (None, 19, 30, 512)  262144      conv_dw_10_relu[0][0]
__________________________________________________________________________________________________
conv_pw_10_bn (BatchNormalizati (None, 19, 30, 512)  2048        conv_pw_10[0][0]
__________________________________________________________________________________________________
conv_pw_10_relu (Activation)    (None, 19, 30, 512)  0           conv_pw_10_bn[0][0]
__________________________________________________________________________________________________
conv_dw_11 (DepthwiseConv2D)    (None, 19, 30, 512)  4608        conv_pw_10_relu[0][0]
__________________________________________________________________________________________________
conv_dw_11_bn (BatchNormalizati (None, 19, 30, 512)  2048        conv_dw_11[0][0]
__________________________________________________________________________________________________
conv_dw_11_relu (Activation)    (None, 19, 30, 512)  0           conv_dw_11_bn[0][0]
__________________________________________________________________________________________________
conv_pw_11 (Conv2D)             (None, 19, 30, 512)  262144      conv_dw_11_relu[0][0]
__________________________________________________________________________________________________
conv_pw_11_bn (BatchNormalizati (None, 19, 30, 512)  2048        conv_pw_11[0][0]
__________________________________________________________________________________________________
conv_pw_11_relu (Activation)    (None, 19, 30, 512)  0           conv_pw_11_bn[0][0]
__________________________________________________________________________________________________
conv_dw_12 (DepthwiseConv2D)    (None, 10, 15, 512)  4608        conv_pw_11_relu[0][0]
__________________________________________________________________________________________________
conv_dw_12_bn (BatchNormalizati (None, 10, 15, 512)  2048        conv_dw_12[0][0]
__________________________________________________________________________________________________
conv_dw_12_relu (Activation)    (None, 10, 15, 512)  0           conv_dw_12_bn[0][0]
__________________________________________________________________________________________________
conv_pw_12 (Conv2D)             (None, 10, 15, 1024) 524288      conv_dw_12_relu[0][0]
__________________________________________________________________________________________________
conv_pw_12_bn (BatchNormalizati (None, 10, 15, 1024) 4096        conv_pw_12[0][0]
__________________________________________________________________________________________________
conv_pw_12_relu (Activation)    (None, 10, 15, 1024) 0           conv_pw_12_bn[0][0]
__________________________________________________________________________________________________
conv_dw_13 (DepthwiseConv2D)    (None, 10, 15, 1024) 9216        conv_pw_12_relu[0][0]
__________________________________________________________________________________________________
conv_dw_13_bn (BatchNormalizati (None, 10, 15, 1024) 4096        conv_dw_13[0][0]
__________________________________________________________________________________________________
conv_dw_13_relu (Activation)    (None, 10, 15, 1024) 0           conv_dw_13_bn[0][0]
__________________________________________________________________________________________________
conv_pw_13 (Conv2D)             (None, 10, 15, 1024) 1048576     conv_dw_13_relu[0][0]
__________________________________________________________________________________________________
conv_pw_13_bn (BatchNormalizati (None, 10, 15, 1024) 4096        conv_pw_13[0][0]
__________________________________________________________________________________________________
conv_pw_13_relu (Activation)    (None, 10, 15, 1024) 0           conv_pw_13_bn[0][0]
__________________________________________________________________________________________________
classes4 (Conv2D)               (None, 19, 30, 18)   82962       conv_pw_10_relu[0][0]
__________________________________________________________________________________________________
classes5 (Conv2D)               (None, 19, 30, 18)   82962       conv_pw_11_relu[0][0]
__________________________________________________________________________________________________
classes6 (Conv2D)               (None, 10, 15, 18)   165906      conv_pw_12_relu[0][0]
__________________________________________________________________________________________________
classes7 (Conv2D)               (None, 10, 15, 18)   165906      conv_pw_13_relu[0][0]
__________________________________________________________________________________________________
box4 (Conv2D)                   (None, 19, 30, 12)   55308       conv_pw_10_relu[0][0]
__________________________________________________________________________________________________
box5 (Conv2D)                   (None, 19, 30, 12)   55308       conv_pw_11_relu[0][0]
__________________________________________________________________________________________________
box6 (Conv2D)                   (None, 10, 15, 12)   110604      conv_pw_12_relu[0][0]
__________________________________________________________________________________________________
box7 (Conv2D)                   (None, 10, 15, 12)   110604      conv_pw_13_relu[0][0]
__________________________________________________________________________________________________
classes4_reshape (Reshape)      (None, 1710, 6)      0           classes4[0][0]
__________________________________________________________________________________________________
classes5_reshape (Reshape)      (None, 1710, 6)      0           classes5[0][0]
__________________________________________________________________________________________________
classes6_reshape (Reshape)      (None, 450, 6)       0           classes6[0][0]
__________________________________________________________________________________________________
classes7_reshape (Reshape)      (None, 450, 6)       0           classes7[0][0]
__________________________________________________________________________________________________
anchors4 (AnchorBoxes)          (None, 19, 30, 3, 8) 0           box4[0][0]
__________________________________________________________________________________________________
anchors5 (AnchorBoxes)          (None, 19, 30, 3, 8) 0           box5[0][0]
__________________________________________________________________________________________________
anchors6 (AnchorBoxes)          (None, 10, 15, 3, 8) 0           box6[0][0]
__________________________________________________________________________________________________
anchors7 (AnchorBoxes)          (None, 10, 15, 3, 8) 0           box7[0][0]
__________________________________________________________________________________________________
concatenate_classes (Concatenat (None, 4320, 6)      0           classes4_reshape[0][0]
                                                                 classes5_reshape[0][0]
                                                                 classes6_reshape[0][0]
                                                                 classes7_reshape[0][0]
__________________________________________________________________________________________________
boxes4_reshape (Reshape)        (None, 1710, 4)      0           box4[0][0]
__________________________________________________________________________________________________
boxes5_reshape (Reshape)        (None, 1710, 4)      0           box5[0][0]
__________________________________________________________________________________________________
boxes6_reshape (Reshape)        (None, 450, 4)       0           box6[0][0]
__________________________________________________________________________________________________
boxes7_reshape (Reshape)        (None, 450, 4)       0           box7[0][0]
__________________________________________________________________________________________________
anchors4_reshape (Reshape)      (None, 1710, 8)      0           anchors4[0][0]
__________________________________________________________________________________________________
anchors5_reshape (Reshape)      (None, 1710, 8)      0           anchors5[0][0]
__________________________________________________________________________________________________
anchors6_reshape (Reshape)      (None, 450, 8)       0           anchors6[0][0]
__________________________________________________________________________________________________
anchors7_reshape (Reshape)      (None, 450, 8)       0           anchors7[0][0]
__________________________________________________________________________________________________
classes_softmax (Activation)    (None, 4320, 6)      0           concatenate_classes[0][0]
__________________________________________________________________________________________________
concatenate_boxes (Concatenate) (None, 4320, 4)      0           boxes4_reshape[0][0]
                                                                 boxes5_reshape[0][0]
                                                                 boxes6_reshape[0][0]
                                                                 boxes7_reshape[0][0]
__________________________________________________________________________________________________
concatenate_anchors (Concatenat (None, 4320, 8)      0           anchors4_reshape[0][0]
                                                                 anchors5_reshape[0][0]
                                                                 anchors6_reshape[0][0]
                                                                 anchors7_reshape[0][0]
__________________________________________________________________________________________________
concatenate_output (Concatenate (None, 4320, 18)     0           classes_softmax[0][0]
                                                                 concatenate_boxes[0][0]
                                                                 concatenate_anchors[0][0]
==================================================================================================
Total params: 4,058,424
Trainable params: 4,036,536
Non-trainable params: 21,888
__________________________________________________________________________________________________

```