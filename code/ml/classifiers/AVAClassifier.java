package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.data.DataSet;
import ml.data.Example;

/**
 * All-Versus-All (AVA) multiclass classifier.
 * @author Bleecker Coyne
 */
public class AVAClassifier implements Classifier {

    /** Factory that builds a fresh binary classifier per pair. */
    private final ClassifierFactory factory;

    /** Deterministic list of labels present in training data. */
    private List<Double> labelList;

    /** Pairwise classifiers: key "i|j" -> trained binary classifier for (i, j) with i < j. */
    private final Map<String, Classifier> pairwise = new HashMap<>();

    /**
     * Constructor
     * @param factory
     */
    public AVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Key for unordered pair (a, b) with a <= b
     * @param a 
     * @param b 
     * @return string key "a|b"
     */
    private String key(double a, double b) {
        double i = Math.min(a, b);
        double j = Math.max(a, b);
        return i + "|" + j;
    }

    /**
     * Tie-breaking rule: prefer smaller numeric label in case of tie.
     * @param candidate 
     * @param incumbent 
     * @return true if candidate is preferred to incumbent
     */
    private boolean tieBreak(Double candidate, Double incumbent) {
        if (incumbent == null)
            return true;
        return candidate < incumbent;
    }

    /**
     * Helper to return smallest label in labelList, or 0 if labelList is empty or null.
     * @return smallest label or 0
     */
    private double smallestLabel() {
        if (labelList != null && !labelList.isEmpty()) {
            double min = labelList.get(0);
            for (double v : labelList)
                min = Math.min(min, v);
            return min;
        }
        return 0.0;
    }

    /**
     * Train AVA classifier on data.
     * @param data Training data
     */
    @Override
    public void train(DataSet data) {
        labelList = new ArrayList<>(data.getLabels());

        Map<Double, List<Example>> byLabel = new HashMap<>();
        for (Double label : labelList)
            byLabel.put(label, new ArrayList<>());
        for (Example example : data.getData()) {
            byLabel.get(example.getLabel()).add(example);
        }

        // Build a classifier for every unordered pair (i, j) with i < j
        for (int a = 0; a < labelList.size(); a++) {
            double i = labelList.get(a);
            for (int b = a + 1; b < labelList.size(); b++) {
                double j = labelList.get(b);

                // Build pair dataset by concatenating the two buckets, relabeling
                DataSet binary = new DataSet(data.getFeatureMap());

                // i -> +1
                for (Example example : byLabel.get(i)) {
                    Example copy = new Example(example);
                    copy.setLabel(1.0);
                    binary.addData(copy);
                }
                // j -> -1
                for (Example example : byLabel.get(j)) {
                    Example copy = new Example(example);
                    copy.setLabel(-1.0);
                    binary.addData(copy);
                }

                Classifier clf = factory.getClassifier();
                clf.train(binary);
                pairwise.put(key(i, j), clf);
            }
        }
    }

    /**
     * Classify example by weighted voting among pairwise classifiers.
     * @param example Example to classify
     * @return Predicted label
     */
    @Override
    public double classify(Example example) {
        Map<Double, Double> score = new HashMap<>();
        for (double label : labelList) {
            score.put(label, 0.0);
        }

        for (int a = 0; a < labelList.size(); a++) {
            double i = labelList.get(a);
            for (int b = a + 1; b < labelList.size(); b++) {
                double j = labelList.get(b);

                Classifier clf = pairwise.get(key(i, j));

                double prediction = clf.classify(example); 
                double confidence = clf.confidence(example); 

                double y = (prediction == 1.0) ? +confidence : -confidence;
                score.put(i, score.get(i) + y);
                score.put(j, score.get(j) - y);
            }
        }
        Double bestLabel = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (double label : labelList) {
            double labelScore = score.get(label);
            if (labelScore > bestScore || (labelScore == bestScore && tieBreak(label, bestLabel))) {
                bestScore = labelScore;
                bestLabel = label;
            }
        }
        return (bestLabel != null) ? bestLabel : smallestLabel();
    }

    @Override
    public double confidence(Example example) {
        return 0.0;
    }
}