package ml.classifiers;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Experiments for Assignment 8, Question 3
 * 10-fold cross validation with varying number of hidden nodes
 * @author Bleecker Coyne
 */
public class ExperimentsP3 {
    
    private static final String TITANIC_TRAIN = "/Users/bleeckercoyne/Desktop/CS 158/assign-8-bmcoyne/data/titanic-train.csv";
    private static final double ETA = 0.1;  // default
    private static final int ITERATIONS = 200;  // default
    private static final int NUM_FOLDS = 10;

    public static void main(String[] args) {
        DataSet fullData = new DataSet(TITANIC_TRAIN, DataSet.CSVFILE);
        // Create 10-fold cross validation
        CrossValidationSet cvSet = fullData.getRandomCrossValidationSet(NUM_FOLDS); 
        // Print header row
        System.out.println("Hidden nodes\tTraining average\tTesting average");

        for (int hiddenNodes = 1; hiddenNodes <= 10; hiddenNodes++) {
            double totalTrainAccuracy = 0.0;
            double totalTestAccuracy = 0.0;
            // Run 10-fold cross validation for this number of hidden nodes
            for (int fold = 0; fold < NUM_FOLDS; fold++) {
                DataSetSplit split = cvSet.getValidationSet(fold);
                DataSet trainData = split.getTrain();
                DataSet testData = split.getTest();
                
                TwoLayerNN nn = new TwoLayerNN(hiddenNodes);
                nn.setEta(ETA);
                nn.setIterations(ITERATIONS);
                nn.train(trainData);
                
                double trainAccuracy = calculateAccuracy(nn, trainData);
                totalTrainAccuracy += trainAccuracy;
                double testAccuracy = calculateAccuracy(nn, testData);
                totalTestAccuracy += testAccuracy;
            }
            
            // Calculate averages across all folds
            double avgTrainAccuracy = totalTrainAccuracy / NUM_FOLDS;
            double avgTestAccuracy = totalTestAccuracy / NUM_FOLDS;
            
            // Print results for this number of hidden nodes
            System.out.printf("%d\t%.6f\t%.6f%n", 
                hiddenNodes, avgTrainAccuracy, avgTestAccuracy);
        }
    }
    
    /**
     * Calculate accuracy on a dataset
     * @param nn The neural network
     * @param data The dataset
     * @return The accuracy as a double
     */
    private static double calculateAccuracy(TwoLayerNN nn, DataSet data) {
        int correct = 0;
        int total = 0;
        
        for (Example example : data.getData()) {
            double prediction = nn.classify(example);
            double label = example.getLabel();
            
            if (prediction == label) {
                correct++;
            }
            total++;
        }
        
        return (double) correct / total;
    }
}