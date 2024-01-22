from KNN_class import KNN
def main(filename):
    print(f"File name received as input: {filename}")
    knn1 = KNN(14,'ResNet','Manhattan')
    knn1.printParmeters()
    knn1.loadDataset(filename)
    knn1.splitDataset(0.75)
    knn1.predict_vectorized()
    knn1.calc_performance()
    knn1.printPerformance()
    
    


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Please Enter file-name as parameter")
        sys.exit(1)

    filename = sys.argv[1]
    main(filename)
