package ml.classifiers;

import ml.data.Example;
import ml.data.DataSet;
import java.util.*;

/**
 * CS158 - Assignment 3
 * Perceptron Classifier
 * 
 * @author: Bleecekr Coyne
 * 
 * This is my implementation of the Basic Perceptron learning algorithm.
 * 
 */

 /**
  * The PerceptronClassifier class implements the Classifier interface and provides methods to train
  * a perceptron model on a given dataset and classify new examples based on the learned weights.
  * 
  */
public class PerceptronClassifier implements Classifier {

    private DataSet dataset;
    private List<Example> examples;
    protected HashMap<Integer, Double> weights; 
    private Double bias; // bias
    private List<Integer> features;
    private Integer maxIterations;
    private Random rand = new Random();

    /**
     * Zero-parameter constructor to initialize the PerceptronClassifier with default values.
     */
    public PerceptronClassifier() {
        this.weights = new HashMap<Integer, Double>();
        this.bias = 0.0;
        this.maxIterations = 10; // default
    }

    /**
     * Set the maximum number of iterations for training the perceptron.   
     * @param maxIterations
     */
    public void setIterations(Integer maxIterations) {
        this.maxIterations = maxIterations;
    }

    /**
     * Shuffle the examples in the dataset to ensure random order during training.
     * @param examples the list of examples to be shuffled 
     */
    private void shuffleExamples(List<Example> examples) {
        Collections.shuffle(examples, rand);
    }

    /**
     * Calculate the weighted sum of the features for a given example, using the provided weights and bias.
     * @param example the example for which to calculate the weighted sum
     * @param weight  the weights associated with each feature
     * @param bias the bias term
     * @return the weighted sum
     */
    private double weightedSum(Example example, HashMap<Integer, Double> weight, Double bias) {
        double sum = bias;
        for (Integer feature : example.getFeatureSet()) {
            sum += weight.get(feature) * example.getFeature(feature);
        }
        return sum;
    }

    /**
     * Check if the given example is misclassified based on the current weights and bias.
     * @param example the example to be checked
     * @param weight the weights associated with each feature
     * @param bias the bias term
     * @return true if the example is misclassified, false otherwise
     */ 
    private boolean misclassified(Example example, HashMap<Integer, Double> weight, Double bias) {
        return weightedSum(example, weights, bias) * example.getLabel() <= 0;
    }

    /**
     * Update the weights based on the features of the misclassified example.
     * @param weights the current weights to be updated
     * @param example the misclassified example used for updating the weights
     */
    private void updatedWeights(HashMap<Integer, Double> weights, Example example) {
        double label = example.getLabel();
        for (Integer featureIndex : weights.keySet()) {
            double oldW = weights.get(featureIndex);
            double newW = example.getFeature(featureIndex);
            weights.put(featureIndex, oldW + label * newW);
        }
    }

    /**
     * Train the perceptron model using the provided examples, features, and maximum iterations.
     * @param examples the list of training examples
     * @param features the list of feature indices
     * @param maxIterations the maximum number of iterations for training
     */
    private void perceptronTrain(List<Example> examples, List<Integer> features, Integer maxIterations) { // inputs
        for (int i = 0; i < maxIterations; i++) {
            shuffleExamples(examples); // randomize order of features
            for (Example example : examples) {
                if (misclassified(example, weights, bias)) {
                    updatedWeights(weights, example);
                }
            }
        }
    }

    /**
     * Train the perceptron classifier on the given dataset.
     * @param dataset the dataset used for training
     */
    @Override
    public void train(DataSet dataset) {
        List<Integer> features = new ArrayList<>(dataset.getAllFeatureIndices());
        List<Example> examples = dataset.getData();
        this.dataset = dataset;
        this.examples = examples;
        this.features = features;
        for (Integer j : this.features) {
            this.weights.put(j, 0.0);
        }
        perceptronTrain(examples, features, maxIterations);
    }

    /**
     * Classify a given example using the trained perceptron model.
     * @param ex the example to be classified
     */
    @Override
    public double classify(Example ex) {
        double s = weightedSum(ex, weights, bias);
        return s >= 0 ? 1.0 : -1.0; // must return ±1
    }

    /**
     * Return a string representation of the perceptron model, including weights and bias.
     * @return a string representation of the perceptron model
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        List<Integer> sorted = new ArrayList<>(features);
        Collections.sort(sorted);
        for (int k = 0; k < sorted.size(); k++) {
            int j = sorted.get(k);
            sb.append(j).append(":").append(weights.get(j));
            if (k < sorted.size() - 1)
                sb.append(' ');
        }
        sb.append(' ').append(bias);
        return sb.toString();
    }

    @Override
    public double confidence(Example example) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'confidence'");
    }
}
