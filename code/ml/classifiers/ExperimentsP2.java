package ml.classifiers;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Experiments for Assignment 8, Question 2
 * Tracks loss, training accuracy, and testing accuracy over iterations
 * 
 * @author Bleecker Coyne
 */
public class ExperimentsP2 {

    private static final String TITANIC_TRAIN = "/Users/bleeckercoyne/Desktop/CS 158/assign-8-bmcoyne/data/titanic-train.csv";
    private static final int HIDDEN_NODES = 3;
    private static final double ETA = 0.1;
    private static final int ITERATIONS = 200;

    public static void main(String[] args) {
        DataSet fullData = new DataSet(TITANIC_TRAIN, DataSet.CSVFILE);
        DataSetSplit split = fullData.split(0.9);
        final DataSet trainData = split.getTrain();
        final DataSet testData = split.getTest();

        TwoLayerNN nn = new TwoLayerNN(HIDDEN_NODES);
        nn.setEta(ETA);
        nn.setIterations(ITERATIONS);

        // Print header row
        System.out.println("Iteration\tSum of squared errors\tTraining accuracy\tTesting accuracy");

        // Train with callback to track metrics at each iteration
        nn.trainWithTracking(trainData, new TwoLayerNN.TrainingCallback() {
            @Override
            public void onIterationComplete(int iteration, double squaredError) {
                double trainAccuracy = calculateAccuracy(nn, trainData);
                double testAccuracy = calculateAccuracy(nn, testData);

                // Print metrics for this iteration
                System.out.printf("%d\t%.6f\t%.6f\t%.6f%n",
                        iteration, squaredError, trainAccuracy, testAccuracy);
            }
            /**
             * Calculate accuracy on a dataset
             * @param nn The neural network
             * @param data The dataset
             * @return The accuracy as a double between 0 and 1
             */
            private double calculateAccuracy(TwoLayerNN nn, DataSet data) {
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
        });
    }
}