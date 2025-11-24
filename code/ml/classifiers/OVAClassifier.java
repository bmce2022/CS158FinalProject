package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.data.DataSet;
import ml.data.Example;

/**
 * One-Versus-All (OVA) multiclass classifier.
 * @author Bleecker Coyne
 */

public class OVAClassifier implements Classifier{
    // One-vs-All Classifier

    /** Factory that builds a fresh binary classifier for each label. */
    private final ClassifierFactory factory;

    /** One-vs-all classifiers: original multiclass label -> trained binary classifier. */
    private final Map<Double, Classifier> oneVsAll = new HashMap<>();

    /** list of labels from the training data */
    private List<Double> labelList;

    /**
     * Constructor
     * @param factory
     */
    public OVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Tie-breaking rule: prefer smaller numeric label in case of tie.
     * @param candidate
     * @param incumbent
     * @return true if candidate is preferred to incumbent
     */
    private boolean tieBreak(Double candidate, Double incumbent) {
        if (incumbent == null) return true;
        return candidate < incumbent;
    }

    /**
     * Helper to return smallest label in labelList, or 0 if labelList is empty or null.
     * @return smallest label or 0
     */
    private double smallestLabel() {
        // return smallest label if available, else 0
        if (labelList != null && !labelList.isEmpty()) {
            double min = labelList.get(0);
            for (double v : labelList) min = Math.min(min, v);
            return min;
        }
        return 0.0;
    }

    /**
     * Train the OVA meta-classifier.
     * @param data Training data
     */
    @Override
    public void train(DataSet data) {
        labelList = new ArrayList<>(data.getLabels());

        // Pre-index examples by label (one pass over data)
        Map<Double, List<Example>> byLabel = new HashMap<>();
        for (Double label : labelList) byLabel.put(label, new ArrayList<>());
        for (Example example : data.getData()) {
            byLabel.get(example.getLabel()).add(example);
        }
    
        // For each label, build +1/-1 dataset from the buckets and train
        for (Double label : labelList) {
            DataSet binaryData = new DataSet(data.getFeatureMap());

            for (Example example : byLabel.get(label)) {
                Example copy = new Example(example);
                copy.setLabel(1.0);
                binaryData.addData(copy);
            }

            for (Double other : labelList) {
                if (!other.equals(label)) {
                    for (Example example : byLabel.get(other)) {
                        Example copy = new Example(example);
                        copy.setLabel(-1.0);
                        binaryData.addData(copy);
                    }
                }
            }
    
            Classifier newClassifier = factory.getClassifier();
            newClassifier.train(binaryData);
            oneVsAll.put(label, newClassifier);
        }
    }

    /**
     * Classify an example using the OVA strategy.
     * @param example Example to classify
     * @return Predicted label
     */
    @Override
    public double classify(Example example) {
        Double bestPositiveLabel = null;
        double bestPositiveConfidenceScore = Double.NEGATIVE_INFINITY;

        Double leastNegLabel = null;
        double leastNegConfidenceScore = Double.POSITIVE_INFINITY;

        // Evaluate all per-label binary classifiers
        for (Map.Entry<Double, Classifier> entry : oneVsAll.entrySet()) {
            Double label = entry.getKey();
            Classifier clf = entry.getValue();

            double prediction = clf.classify(example);    
            double confidence = clf.confidence(example); 

            if (prediction == 1.0) {
                // Keep the most confident positive
                if (confidence > bestPositiveConfidenceScore || (confidence == bestPositiveConfidenceScore && tieBreak(label, bestPositiveLabel))) {
                    bestPositiveConfidenceScore = confidence;
                    bestPositiveLabel = label;
                }
            } else {
                // Track the least confident negative (smallest confidence)
                if (confidence < leastNegConfidenceScore || (confidence == leastNegConfidenceScore && tieBreak(label, leastNegLabel))) {
                    leastNegConfidenceScore = confidence;
                    leastNegLabel = label;
                }
            }
        }

        if (bestPositiveLabel != null) {
            return bestPositiveLabel;
        } else {
            return (leastNegLabel != null) ? leastNegLabel : smallestLabel();
        }
    }

    /**
     * OVA meta-classifier confidence is unspecified in the assignment;
     * return 0 as allowed.
     */
    @Override
    public double confidence(Example example) {
        return 0.0;
    }
}
