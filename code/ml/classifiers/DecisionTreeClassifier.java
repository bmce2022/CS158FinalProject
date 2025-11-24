package ml.classifiers;
import ml.data.Example;
import ml.data.DataSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * CS158 - Assignment 2
 * Decision Trees
 * 
 * @author: Bleecekr Coyne
 * 
 * This file implements a DecisionTreeClassifier that uses a top-down
 * recursive algorithm to build a decision tree for binary classification.
 * 
 * Requirements:
 * - Implements the Classifier interface.
 * - Supports a depth limit for early stopping.
 * - Uses minimum training error to choose splits.
 * - Uses DecisionTreeNode to represent the tree structure.
 */


 /**
 * My Decision Tree Classifier implementation for binary features.
 * 
 * The tree is trained using recursive partitioning, choosing the feature
 * with the minimum training error at each node. The left branch corresponds
 * to feature value 0.0, and the right branch corresponds to feature value 1.0.
 */
public class DecisionTreeClassifier implements Classifier {

    private DecisionTreeNode root;
    private int depthLimit;
    private DataSet dataset;
    private List<Example> examples;
    private HashMap<Integer, String> featureMap;

     /**
     * Zero-parameter constructor.
     * Initializes an empty DecisionTreeClassifier.
     */
    public DecisionTreeClassifier() {
        this.root = null;
        this.depthLimit = -1; // default to no limit
    }

    /**
     * Set the maximum depth of the tree.
     * 
     * @param depth the maximum depth (0 = single leaf, -1 = unlimited depth)
     */
    public void setDepthLimit(int depth) {
        this.depthLimit = depth;
    }

    /**
     * Select the best feature to split on based on training error.
     * 
     * @param examples training examples
     * @param features data set features
     * @return feature with lowest error
     */
    private int featureSplit(List<Example> examples, List<Integer> features) {
        int bestFeatureIndex = 0; 
        int bestFeatureId = features.get(bestFeatureIndex);
        double bestFeatureError = examples.size() + 1; // start with very high error

        for (int i = 0; i < features.size(); i++) {
            int featureIndex = i; 
            int featureId = features.get(i); 
            double error = findError(examples, featureIndex);
            if (error < bestFeatureError ) {
                bestFeatureError = error;
                bestFeatureIndex = featureIndex;
                bestFeatureId = featureId;
            }
            else if (error == bestFeatureError) {
                if (featureIndex < bestFeatureIndex){
                    bestFeatureIndex = featureIndex;
                    bestFeatureId = featureId;
                }
            }
        }
        return bestFeatureId;
    }

    /**
     * Compute the error of splitting on a given feature.
     * 0.0 -> left; 1.0 -> right
     * 
     * @param examples training examples
     * @param featureIndex feature index
     * @return error count
     */
    private double findError(List<Example> examples, int featureIndex) {
        List<Example> left = new ArrayList<>();
        List<Example> right = new ArrayList<>();

        for (Example ex : examples) {

            double featureId = ex.getFeature(featureIndex);
            if (featureId == 0.0) {
                left.add(ex);
            } else {
                right.add(ex);
            }
        }

        return sideError(left) + sideError(right);
    }

    /**
     * Compute the error on one side of a split.
     * 
     * @param side examples on one side of the split
     * @return error count
     */
    private double sideError(List<Example> side) {

        if (side.isEmpty())
            return 0.0; // a leaf

        int c0 = 0; // no
        int c1 = 0; // yes
        for (int i = 0; i < side.size(); i++) {

            if (side.get(i).getLabel() == 1.0) {
                c1++;
            } else {
                c0++;
            }

        }
        if (c0 >= c1) {
            return (double) c1;

        } else {
            return (double) c0;
        }
    }

    /**
     * Check if all examples have the same label.
     * 
     * @param examples training examples
     * @return true if all labels are identical
     */
    private boolean sameLabel(List<Example> examples) {
        if (examples.isEmpty())
            return true;

        double firstLabel = examples.get(0).getLabel();
        for (int i = 0; i < examples.size(); i++) {
            if (examples.get(i).getLabel() != firstLabel) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if all examples have identical feature values.
     * 
     * @param examples training examples
     * @return true if all features identical
     */
    private boolean allSameFeatures(DataSet data) {

        int numFeatures = data.getAllFeatureIndices().size();

        for (int f = 0; f < numFeatures; f++) {
            double firstValue = examples.get(0).getFeature(f);
            for (int i = 0; i < examples.size(); i++) {
                if (examples.get(i).getFeature(f) != firstValue) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Find the majority label in the examples.
     * 
     * @param examples training examples
     * @param parentLabel label of the parent node
     * @return majority label
     */
    private double majorityLabel(List<Example> examples, double parentLabel) {
        if (examples.isEmpty())
            return parentLabel; // use parent's label if no examples

        int c0 = 0;
        int c1 = 0;

        for (int i = 0; i < examples.size(); i++) {
            if (examples.get(i).getLabel() == -1.0) {
                c0++;
            } else {
                c1++;
            }
        }

        if (c0 > c1) {
            return -1.0;
        } else if (c1 > c0) {
            return 1.0;
        } else {
            return parentLabel; // return parent's label in case of tie
        }
    }

    /**
     * Recursively builds the decision tree.
     * 
     * @param examples list of training examples
     * @param features available features
     * @param currentDepth current depth in the tree
     * @param parentLabel label of the parent node
     * @return root node of the subtree
     */

    private DecisionTreeNode buildTree(List<Example> examples, List<Integer> features, int currentDepth, double parentLabel) {
        // --- Base cases start
        double majority = majorityLabel(examples, parentLabel);
        if (examples.isEmpty()) {
            return new DecisionTreeNode(majority);
        }
        if (sameLabel(examples)) {
            return new DecisionTreeNode(majority);
        }
        if (features.isEmpty()) {
            return new DecisionTreeNode(majority);
        }
        if (depthLimit != -1 && currentDepth >= depthLimit) {
            return new DecisionTreeNode(majority);
        }
        if (allSameFeatures(dataset)) { 
            return new DecisionTreeNode(majority);
        }
        // --- Base cases end

        // --- Choose best feature to split on
        int featureSplit = featureSplit(examples, features);

        // Create internal node for that feature
        DecisionTreeNode node = new DecisionTreeNode(featureSplit);

        List<Example> leftExamples = new ArrayList<>();
        List<Example> rightExamples = new ArrayList<>();
        for (Example ex : examples) {
            double featureVal = ex.getFeature(featureSplit);
            if (featureVal == 0.0) {
                leftExamples.add(ex);
            } else {
                rightExamples.add(ex);
            }
        }

        // --- Copy feature list for each child; remove used feature 
        List<Integer> leftFeatures = new ArrayList<>(features);
        List<Integer> rightFeatures = new ArrayList<>(features);
        leftFeatures.remove(Integer.valueOf(featureSplit));
        rightFeatures.remove(Integer.valueOf(featureSplit));

        // This node’s majority becomes the parent majority for both children
        double nextParent = majority;

        // --- Recurse (if empty child use leaf with this node’s majority)
        if (leftExamples.isEmpty()) {
            node.setLeft(new DecisionTreeNode(nextParent));
        } else {
            node.setLeft(buildTree(leftExamples, leftFeatures, currentDepth + 1, nextParent));
        }

        if (rightExamples.isEmpty()) {
            node.setRight(new DecisionTreeNode(nextParent));
        } else {
            node.setRight(buildTree(rightExamples, rightFeatures, currentDepth + 1, nextParent));
        }

        return node;
    }

    /**
     * Train the decision tree on the provided dataset.
     * 
     * @param dataset the dataset to train on
     */
    @Override
    public void train(DataSet dataset) {
        List<Integer> features = new ArrayList<>(dataset.getAllFeatureIndices());
        List<Example> examples = dataset.getData();
        this.dataset = dataset;
        this.examples = examples;
        this.featureMap = dataset.getFeatureMap();
        double parentLabel = 0.0; // default parent label to 0.0
        // build decision tree here
        root = buildTree(examples, features, 0, parentLabel);
    }

    /**
     * Classify an example by traversing the trained decision tree.
     * 
     * @param example the example to classify
     * @return predicted label
     */
    @Override
    public double classify(Example example) {

        if (root == null) {
            return 0; // default if tree not trained
        }

        DecisionTreeNode current = root;
        while (!current.isLeaf()) {
            int featureIndex = current.getFeatureIndex();
            double featureValue = example.getFeature(featureIndex);

            if (featureValue == 0.0) {
                current = current.getLeft();
            } else {
                current = current.getRight();
            }
        }

        return (double) current.prediction();
    }

    /**
     * Return a string representation of the tree.
     * 
     * @return formatted tree string
     */
    @Override
    public String toString() {
        HashMap<Integer, String> featureMap = dataset.getFeatureMap();

        if (root == null) {
            return "Empty tree";
        }
        return root.treeString(featureMap);
    }

    @Override
    public double confidence(Example example) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'confidence'");
    }
}
