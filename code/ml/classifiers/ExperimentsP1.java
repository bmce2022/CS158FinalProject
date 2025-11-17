package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.lang.reflect.Field;
import java.util.HashMap;

/**
 * Experiments for Question 1
 * Tests the network from Figure 1 with specific input
 * @author Bleecker Coyne
 */
public class ExperimentsP1 {

    private static final double[][] FIG1_HIDDEN = {
        { -0.5,  1.5, -2.0 },   // into hidden node 1: x1, x2, bias
        {  0.1,  0.5, -1.5 }    // into hidden node 2: x1, x2, bias
    };
    private static final double[] FIG1_OUTPUT = { -1.2, -0.5, 2.0 }; // v1, v2, bias
    
    private static final double ETA = 0.5;
    private static final double X1  = 0.5;
    private static final double X2  = 0.2;
    private static final double Y   = -1.0;

    public static void main(String[] args) throws Exception {

        // Create a dataset with ONLY the single example [0.5, 0.2]
        HashMap<Integer, String> featureMap = new HashMap<>();
        featureMap.put(0, "x1");
        featureMap.put(1, "x2");
        DataSet ds = new DataSet(featureMap);
        
        Example ex = new Example();
        ex.setLabel(Y);
        ex.setFeature(0, X1);   // x1 = 0.5
        ex.setFeature(1, X2);   // x2 = 0.2
        ds.addData(ex);

        // Create neural network with 2 hidden nodes
        TwoLayerNN nn = new TwoLayerNN(2);
        nn.setEta(ETA);
        nn.setIterations(1);  // Only 1 iteration
        nn.setQuestion1(true);
        // Set up biasedDataset and numFeatures manually
        Field biasedField = TwoLayerNN.class.getDeclaredField("biasedDataset");
        biasedField.setAccessible(true);
        DataSet biasedDs = ds.getCopyWithBias();
        biasedField.set(nn, biasedDs);

        Field nfField = TwoLayerNN.class.getDeclaredField("numFeatures");
        nfField.setAccessible(true);
        nfField.set(nn, biasedDs.getAllFeatureIndices().size());

        Field hField = TwoLayerNN.class.getDeclaredField("hiddenWeights");
        Field oField = TwoLayerNN.class.getDeclaredField("outputWeights");
        Field nh     = TwoLayerNN.class.getDeclaredField("hiddenNodes");
        
        hField.setAccessible(true);
        oField.setAccessible(true);
        nh.setAccessible(true);

        // The following were CoPilot's help since my code wasn't running
        @SuppressWarnings("unchecked")
        HashMap<Integer, Double> hW = (HashMap<Integer, Double>) hField.get(nn);
        @SuppressWarnings("unchecked")
        HashMap<Integer, Double> oW = (HashMap<Integer, Double>) oField.get(nn);
        int numFeatures = (int) nfField.get(nn); 
        int hiddenNodes = (int) nh.get(nn); 
        hW.clear();
        oW.clear();

        // Set hidden weights: key = h * numFeatures + f
        for (int h = 0; h < hiddenNodes; h++) {
            for (int f = 0; f < numFeatures; f++) {
                hW.put(h * numFeatures + f, FIG1_HIDDEN[h][f]);
            }
        }
        
        // Set output weights: v1, v2 at indices 0,1 and bias at index hiddenNodes
        for (int h = 0; h < hiddenNodes; h++) {
            oW.put(h, FIG1_OUTPUT[h]);
        }
        oW.put(hiddenNodes, FIG1_OUTPUT[2]); 

        System.out.println("Initial weights set to Figure 1 values:");
        System.out.println("Hidden layer:");
        System.out.println("  h1: x1=" + FIG1_HIDDEN[0][0] + ", x2=" + FIG1_HIDDEN[0][1] + ", bias=" + FIG1_HIDDEN[0][2]);
        System.out.println("  h2: x1=" + FIG1_HIDDEN[1][0] + ", x2=" + FIG1_HIDDEN[1][1] + ", bias=" + FIG1_HIDDEN[1][2]);
        System.out.println("Output layer:");
        System.out.println("  v1=" + FIG1_OUTPUT[0] + ", v2=" + FIG1_OUTPUT[1] + ", bias=" + FIG1_OUTPUT[2]);
        System.out.println();

        Example biasedEx = biasedDs.getData().get(0);
        
        // Use reflection to call the private forward and backward pass methods
        java.lang.reflect.Method forwardMethod = TwoLayerNN.class.getDeclaredMethod("forwardPass", Example.class);
        java.lang.reflect.Method backwardMethod = TwoLayerNN.class.getDeclaredMethod("backwardPass", Example.class, double.class, double.class);
        forwardMethod.setAccessible(true);
        backwardMethod.setAccessible(true);
        
        // Do forward pass 
        double prediction = (double) forwardMethod.invoke(nn, biasedEx);
        
        // Do backward pass 
        backwardMethod.invoke(nn, biasedEx, prediction, Y);
        
        System.out.println("=======================================================");
        System.out.println("Done!");
        System.out.println("=======================================================");
    }
}