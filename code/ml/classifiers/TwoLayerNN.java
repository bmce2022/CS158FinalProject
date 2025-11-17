package ml.classifiers;

import java.util.*;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Two-layer Neural Network Classifier
 * - Input layer to hidden layer with tanh activation
 * - Hidden layer to output layer with tanh activation
 * - Trained with backpropagation
 * 
 * @author Bleecker Coyne
 */
public class TwoLayerNN implements Classifier {

    private int hiddenNodes;
    private double learningRate = 0.1;
    private int iterations = 200;
    private HashMap<Integer, Double> hiddenWeights = new HashMap<Integer, Double>(); // weights from input to hidden
    private HashMap<Integer, Double> outputWeights = new HashMap<Integer, Double>(); // weights from hidden to output
    private double[] hiddenSums = new double[hiddenNodes];
    private double[] hiddenActivations = new double[hiddenNodes];
    private double outputSum;
    private int numFeatures;
    private DataSet biasedDataset;
    private Boolean question1print = false;

    // for following are hardcoded for normalized baseball card data
    private double minPrice = 108.49;
    private double maxPrice = 655.00;

    public TwoLayerNN(int numHiddenNodes) {
        this.hiddenNodes = numHiddenNodes;
        this.hiddenSums = new double[numHiddenNodes];
        this.hiddenActivations = new double[numHiddenNodes];
    }

    /**
     * Set the learning rate (eta)
     * 
     * @param eta
     */
    public void setEta(double eta) {
        this.learningRate = eta;
    }

    /**
     * Set the number of training iterations
     * 
     * @param numIterations
     */
    public void setIterations(int numIterations) {
        this.iterations = numIterations;
    }

    /**
     * Set normalization parameters for price denormalization
     * 
     * @param min minimum price in training data
     * @param max maximum price in training data
     */
    public void setNormalizationParams(double min, double max) {
        this.minPrice = min;
        this.maxPrice = max;
    }

    /**
     * Set whether to print question 1 outputs
     * 
     * @param val
     */
    public void setQuestion1(Boolean val) {
        this.question1print = val;
    }

    /**
     * Hyperbolic tangent activation function
     * 
     * @param x input value
     * @return tanh(x)
     */
    public double tanh(double x) {
        return Math.tanh(x);
    }

    /**
     * Derivative of tanh activation function
     * 
     * @param x input value
     * @return tanh'(x)
     */
    public double tanhDerivative(double x) {
        double tanhX = tanh(x);
        return 1 - tanhX * tanhX;
    }

    /**
     * Get the key for hidden weight mapping
     * 
     * @param h hidden node index
     * @param f feature index
     * @return key for hiddenWeights map
     */
    private int getHiddenKey(int h, int f) {
        return h * numFeatures + f;
    }

    /**
     * Generate random weights for given features
     * 
     * @param features set of feature indices
     * @return map of feature index to random weight
     */
    public HashMap<Integer, Double> randomWeights(Set<Integer> features) {
        HashMap<Integer, Double> weights = new HashMap<Integer, Double>();
        for (Integer feature : features) {
            weights.put(feature, 0.2 * Math.random() - 0.1); // random weight between -1 and 1
        }
        return weights;
    }

    /**
     * Initialize network weights
     * 
     * @param featureCount number of input features
     */
    public void initializeNetwork(int featureCount) {
        this.numFeatures = featureCount;

        for (int h = 0; h < hiddenNodes; h++) {
            for (int f = 0; f < numFeatures; f++) {
                double randomWeight = (Math.random() * 0.2) - 0.1; // random between -0.1 and 0.1
                hiddenWeights.put(getHiddenKey(h, f), randomWeight);
            }
        }

        for (int h = 0; h < hiddenNodes; h++) {
            double randomWeight = (Math.random() * 0.2) - 0.1; // random between -0.1 and 0.1
            outputWeights.put(h, randomWeight);
        }

        outputWeights.put(hiddenNodes, (Math.random() * 0.2) - 0.1); // random between -0.1 and 0.1
    }

    /**
     * Forward pass through the network
     * 
     * @param example input example
     * @return output of the network
     */
    private double forwardPass(Example example) {
        // Compute hidden layer activations
        for (int h = 0; h < hiddenNodes; h++) {
            double sum = 0.0;

            // Sum over all input features
            for (Integer featureIndex : example.getFeatureSet()) {
                double featureValue = example.getFeature(featureIndex);
                double weight = hiddenWeights.get(getHiddenKey(h, featureIndex));
                sum += weight * featureValue;
            }
            hiddenSums[h] = sum;
            hiddenActivations[h] = tanh(sum);
        }

        // Compute output layer activation
        double outSum = outputWeights.get(hiddenNodes);
        for (int h = 0; h < hiddenNodes; h++) {
            outSum += outputWeights.get(h) * hiddenActivations[h];
        }

        outputSum = outSum;

        if (question1print) { // print statement for experiment 1
            System.out.println("1(a) Node outputs:");
            for (int h = 0; h < hiddenNodes; h++) {
                System.out.printf("  h%d = %.6f%n", h + 1, hiddenActivations[h]);
            }
            System.out.printf("  y  = %.6f%n", tanh(outSum));
            System.out.println();
        }

        return tanh(outSum);
    }

    /**
     * Backward pass to update weights
     * 
     * @param example    input example
     * @param prediction network prediction
     * @param label
     */
    private void backwardPass(Example example, double prediction, double label) {
        double error = label - prediction; // (y-f(v*h))
        double outputDelta = error * tanhDerivative(outputSum); // (y-f(v*h)) * f'(v*h)
        double[] hiddenDeltas = new double[hiddenNodes]; // hidden layer deltas
        for (int h = 0; h < hiddenNodes; h++) {
            double v_k = outputWeights.get(h); // weight from hidden h to output
            hiddenDeltas[h] = outputDelta * v_k * tanhDerivative(hiddenSums[h]); // (y-f(v*h)) * f'(v*h) * v_k * f'(w_k*
                                                                                 // x)
        }

        for (int h = 0; h < hiddenNodes; h++) {
            double v_k = outputWeights.get(h);
            double newWeight = v_k + learningRate * outputDelta * hiddenActivations[h]; // v_k + eta * (y-f(v*h)) *
                                                                                        // f'(v*h) * h_k
            outputWeights.put(h, newWeight);
        }

        double oldBias = outputWeights.get(hiddenNodes);
        outputWeights.put(hiddenNodes, oldBias + learningRate * outputDelta * 1.0);

        for (int h = 0; h < hiddenNodes; h++) {
            for (Integer featureIndex : example.getFeatureSet()) {
                double featureValue = example.getFeature(featureIndex); // x_i
                double oldWeight = hiddenWeights.get(getHiddenKey(h, featureIndex)); // w_ki
                double newWeight = oldWeight + learningRate * hiddenDeltas[h] * featureValue; // w_ki + eta * (y-f(v*h))
                                                                                              // * f'(v*h) * v_k *
                                                                                              // f'(w_k* x) * x_i
                hiddenWeights.put(getHiddenKey(h, featureIndex), newWeight);
            }
        }

        if (question1print) { // print statement for experiment 1

            System.out.println("1(b) Final weights after one iteration:");
            System.out.println("Hidden weights:");
            System.out.println("x1 x2 bias");
            for (int h = 0; h < hiddenNodes; h++) {
                double w1 = hiddenWeights.get(getHiddenKey(h, 0));
                double w2 = hiddenWeights.get(getHiddenKey(h, 1));
                double wb = hiddenWeights.get(getHiddenKey(h, 2));
                System.out.printf("[% .6f, % .6f, % .6f]%n", w1, w2, wb);
            }

            System.out.println("Output weights:");
            System.out.println("v1 v2 bias");
            double v1 = outputWeights.get(0);
            double v2 = outputWeights.get(1);
            double vb = outputWeights.get(hiddenNodes);
            System.out.printf("[% .6f, % .6f, % .6f]%n", v1, v2, vb);
            System.out.println();
        }
    }

    @Override
    public void train(DataSet data) {
        this.biasedDataset = data.getCopyWithBias();

        // Initialize network with correct number of features (including bias)
        numFeatures = biasedDataset.getAllFeatureIndices().size();
        initializeNetwork(numFeatures);

        List<Example> examples = biasedDataset.getData();

        for (int iter = 0; iter < iterations; iter++) {
            for (Example example : examples) {
                // Forward pass
                double prediction = forwardPass(example);
                double label = example.getLabel();
                // Backward pass
                backwardPass(example, prediction, label);
            }
        }
    }

    /**
     * Train with callback for tracking metrics (for Question 2)
     * 
     * @param data training dataset
     */
    public void trainWithTracking(DataSet data, TrainingCallback callback) {
        this.biasedDataset = data.getCopyWithBias();
        numFeatures = biasedDataset.getAllFeatureIndices().size();
        initializeNetwork(numFeatures);
        List<Example> examples = biasedDataset.getData();

        for (int iter = 0; iter < iterations; iter++) {
            double totalSquaredError = 0.0;
            for (Example example : examples) {
                double prediction = forwardPass(example);
                double label = example.getLabel();
                // Accumulate squared error
                double error = label - prediction;
                totalSquaredError += error * error;
                backwardPass(example, prediction, label);
            }
            // Call the callback with metrics for this iteration
            if (callback != null) {
                callback.onIterationComplete(iter + 1, totalSquaredError);
            }
        }
    }

    /**
     * Callback interface for training tracking
     */
    public interface TrainingCallback {
        void onIterationComplete(int iteration, double squaredError);
    }

    /**
     * Classify an example
     */
    @Override
    public double classify(Example example) {
        Example biasedExample = biasedDataset.addBiasFeature(example);
        double output = forwardPass(biasedExample);
        return output > 0 ? 1.0 : -1.0;
    }

    /**
     * Get confidence of classification
     */
    @Override
    public double confidence(Example example) {
        Example biasedExample = biasedDataset.addBiasFeature(example);
        double output = forwardPass(biasedExample);
        return Math.abs(output);
    }

    /**
     * Predict continuous value for regression (NEW METHOD)
     * Returns the raw network output without thresholding
     * 
     * @param example input example with features
     * @return predicted normalized value (in range of tanh: -1 to 1)
     */
    public double predictValue(Example example) {
        Example biasedExample = biasedDataset.addBiasFeature(example);
        return forwardPass(biasedExample);
    }

    /**
     * Predict actual price (denormalized) for baseball cards (NEW METHOD)
     * 
     * @param example input example with normalized features
     * @return predicted price in dollars
     */
    public double predictPrice(Example example) {
        double normalizedPrediction = predictValue(example);
        // Denormalize: actual_price = (normalized × (max - min)) + min
        return normalizedPrediction * (maxPrice - minPrice) + minPrice;
    }

    /**
     * Calculate Mean Squared Error on a dataset (NEW METHOD)
     * Useful for evaluating model performance
     * 
     * @param data test dataset
     * @return mean squared error
     */
    public double calculateMSE(DataSet data) {
        DataSet biasedData = data.getCopyWithBias();
        List<Example> examples = biasedData.getData();
        double totalSquaredError = 0.0;

        for (Example example : examples) {
            double prediction = forwardPass(example);
            double label = example.getLabel();
            double error = label - prediction;
            totalSquaredError += error * error;
        }

        return totalSquaredError / examples.size();
    }

}
