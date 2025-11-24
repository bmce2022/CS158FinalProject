package ml.classifiers;
import ml.data.DataSet;
import ml.data.Example;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * CS158 - Assignment 4
 * @author Bleecker Coyne
 * 
 * k-Nearest Neighbors (k-NN) Classifier implementation
 * euclideanDistance: computes the Euclidean distance between two examples
 * majorityLabel: determines the majority label among the k nearest neighbors
 * knnPredict: predicts the label for a given example using the k-NN algorithm
 * train: trains the k-NN classifier by storing the training dataset
 * classify: classifies a new example using the trained k-NN classifier
 */
public class KNNClassifier implements Classifier {
    protected DataSet dataset;
    protected List<Example> examples;
    protected List<Double> predictions;
    protected int k; // number of neighbors to consider
    
    public KNNClassifier() {
        this.k = 3; // default value for k
    }


    public void setK(int k) {
        this.k = k;
    }
    /**
     * Compute the Euclidean distance between two examples
     * @param example the first example
     * @param neighbor the second example
     * @return the Euclidean distance between the two examples
     */
    public double euclideanDistance(Example example, Example neighbor) {
        double distanceSquared = 0;
        for (int i = 0; i < example.getFeatureSet().size(); i++) {
            double exampleFeature = example.getFeature(i);
            double neighborFeature = neighbor.getFeature(i);
            distanceSquared += Math.pow(exampleFeature - neighborFeature, 2);
        }
        return (double) Math.sqrt(distanceSquared);
    }

    /**
     * Determine the majority label among the k nearest neighbors
     * @param neighborDistances a map of neighbors and their distances
     * @param k the number of neighbors to consider
     * @return the majority label (1.0 or -1.0)
     */
    private double majorityLabel(HashMap<Example, Double> neighborDistances, int k) {

        List<Map.Entry<Example, Double>> sortedDistances = new ArrayList<>(neighborDistances.entrySet());
        sortedDistances.sort(Map.Entry.comparingByValue());

        int kk = Math.min(k, sortedDistances.size());
        int countLabelPos = 0; // label = 1.0
        int countLabelNeg = 0; // label = -1.0

        for (int i = 0; i < kk; i++) {
            Example neighbor = sortedDistances.get(i).getKey();
            double neighborLabel = neighbor.getLabel();
            if (neighborLabel == 1.0) countLabelPos++;
            else if (neighborLabel == -1.0) countLabelNeg++;
        }
        // tie goes to -1.0 by default
        return countLabelPos > countLabelNeg ? 1.0 : -1.0;
    }

    /**
     * Predict the label for a given example using the k-NN algorithm
     * @param example the example to predict
     * @return the predicted label (1.0 or -1.0)
     */
    private double knnPredict(Example example) {
        HashMap<Example, Double> distances = new HashMap<>();
        for (Example neighbor : examples) {
            if(neighbor == example) continue; // skip self
            double distance = euclideanDistance(example, neighbor);
            distances.put(neighbor, distance);  
        }
        return majorityLabel(distances, k);
    }

    /**
     * Train the k-NN classifier by storing the training dataset
     * @param dataset the training dataset
     */
    @Override
    public void train(DataSet dataset) {
        this.dataset = dataset;
        this.examples = dataset.getData(); 
        this.predictions = new ArrayList<>();
    }

    /**
     * Classify a new example using the trained k-NN classifier
     * @param example the example to classify
     * @return the predicted label (1.0 or -1.0)
     */
    @Override
    public double classify(Example example) {
        if (examples == null || examples.isEmpty()) {
            throw new IllegalStateException("KNNClassifier not trained: no examples loaded.");
        }
        double pred = knnPredict(example);
        if (this.predictions != null) this.predictions.add(pred);
        return pred;
    }


    @Override
    public double confidence(Example example) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'confidence'");
    }
}
