# Thresholding using KDE

Following are the discriptions of the folders and files in the directory

- `1.ipynb` : contains the implementation of KDE (from scratch) as class and its sample use for both 1D as well as 2D sample data (generated there itself)
- `2.ipynb` : Contains partial code from Assignment 2 (thresholding task). Uses KDE class to perform thresholding in images in the `q4_data` folder
- `q4_data` : A directory containing the dataset used in the question. Contains multiple dataframes and images.
- `outputs` : Contains the output(KDE on horizontal axis, KDE on vertical axis, Original Image, Thresholded connected Image) for each image mentioned in last part of this task.
- `q4_output_from_assignment_2` : contains output images from assignment 2
- `Comparison of results from previous assignment` : Contains the comparison of outputs in `outputs`(image on right) folder with the `q4_output_from_assignment_2`(image on left) for various images. 

### Method of approach for task 2

- We try thresholding by finding horizontal lines dividing paragraphs (using KDE on vertical axis and finding local minimas) and finding columns (if any) by performing KDE on horizontal axis and finding local minimas.

- Certain more complex pre-processing is done before applying KDE so that the results are good. For example, we find the size of the smallest word in the document and quantize all the words based on that size and find their tentative frequency (by dividing size of the word by the size of smallest word). Each word is repeated certain number of times in the KDE dataset according to this tentative frequency. 

- We are also eliminating some local minimas we found since they might just be noisy and not depict actual distribution. To do this, we consider a minima only if it has a sharp rise on both sides of it.

- This nullifies the effect of large word lengths distorting the shape of KDE (since if there are only large words in a line the density of that line reduces)

 - After finding such lines, we make sure that a lines connecting 2 boxes never crosses any of these lines. Optimal value of bandwidth for both KDEs is found manually by experimenting with various bandwidths. The optimal bandwidths are: 
    - h_horizontal = 200
    - h_vertical = 75
