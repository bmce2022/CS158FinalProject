package ml.classifiers;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Compare multiple classifiers for baseball card price prediction
 * Tests: Neural Network, KNN, Decision Tree, Perceptron
 * 
 * @author Bleecker Coyne
 */
public class ClassifierComparator {
    
    private static final double MIN_PRICE = 108.49;
    private static final double MAX_PRICE = 655.00;
    private static final String DATA_FILE = "/Users/bleeckercoyne/Desktop/CS 158/CS158FinalProject/data/normalizedbaseballcard.csv";
    private static final String OUTPUT_DIR = "/Users/bleeckercoyne/Desktop/CS 158/CS158FinalProject/data/";
    
    /**
     * Main method to run all classifier comparisons
     */
    public static void main(String[] args) {
        // Load and split data once for fair comparison
        DataSet fullData = new DataSet(DATA_FILE, DataSet.CSVFILE);
        DataSetSplit split = fullData.split(0.8);
        DataSet trainData = split.getTrain();
        DataSet testData = split.getTest();
        
        System.out.println("========================================");
        System.out.println("BASEBALL CARD PRICE PREDICTION");
        System.out.println("Classifier Comparison");
        System.out.println("========================================");
        System.out.println("Training examples: " + trainData.getData().size());
        System.out.println("Test examples: " + testData.getData().size());
        System.out.println("\n");
        
        // Test each classifier
        testNeuralNetwork(trainData, testData);
        testKNN(trainData, testData);
        testDecisionTree(trainData, testData);
        testPerceptron(trainData, testData);
        
        System.out.println("\n========================================");
        System.out.println("COMPARISON COMPLETE");
        System.out.println("========================================");
    }
    
    /**
     * Test Neural Network
     */
    private static void testNeuralNetwork(DataSet trainData, DataSet testData) {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║       NEURAL NETWORK (2-Layer)         ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println("Configuration: 10 hidden nodes, η=0.01, 5000 iterations\n");
        
        TwoLayerNN nn = new TwoLayerNN(10);
        nn.setEta(0.01);
        nn.setIterations(5000);
        nn.setNormalizationParams(MIN_PRICE, MAX_PRICE);
        
        long startTime = System.currentTimeMillis();
        nn.train(trainData);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Training time: " + trainTime + " ms\n");
        
        generateNeuralNetworkTable(nn, testData, "Neural Network");
        saveNeuralNetworkToCSV(nn, testData, OUTPUT_DIR + "predictions_nn.csv");
    }
    
    /**
     * Generate predictions table specifically for Neural Network
     */
    private static void generateNeuralNetworkTable(TwoLayerNN nn, DataSet data, String modelName) {
        // Print header
        System.out.println("│ Example  │ Actual Price  │ Predicted Price │   Error    │ Percent Error │ Squared Error  │");
        
        // Track aggregate metrics
        double totalSquaredError = 0.0;
        double totalAbsoluteError = 0.0;
        double totalPercentError = 0.0;
        double maxAbsoluteError = 0.0;
        
        int count = 1;
        for (Example example : data.getData()) {
            double actualNormalized = example.getLabel();
            double predictedNormalized = nn.predictValue(example);
            
            double actualPrice = denormalize(actualNormalized);
            double predictedPrice = denormalize(predictedNormalized);
            double error = actualPrice - predictedPrice;
            double percentError = Math.abs(error / actualPrice * 100);
            double squaredError = error * error;
            
            System.out.printf("│ %8d │ $%11.2f │ $%14.2f │ $%9.2f │ %12.2f%% │ $%13.2f │%n", 
                count, actualPrice, predictedPrice, error, percentError, squaredError);
            
            // Accumulate metrics
            totalSquaredError += squaredError;
            totalAbsoluteError += Math.abs(error);
            totalPercentError += percentError;
            maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
            
            count++;
        }
        
        // Calculate and display aggregate metrics
        int n = data.getData().size();
        double mse = totalSquaredError / n;
        double rmse = Math.sqrt(mse);
        double mae = totalAbsoluteError / n;
        double mape = totalPercentError / n;
        
        System.out.println("\n=== Performance Metrics (" + modelName + ") ===");
        System.out.println("Mean Squared Error (MSE):           $" + String.format("%,12.2f", mse));
        System.out.println("Root Mean Squared Error (RMSE):     $" + String.format("%,12.2f", rmse));
        System.out.println("Mean Absolute Error (MAE):          $" + String.format("%,12.2f", mae));
        System.out.println("Mean Absolute Percent Error (MAPE): " + String.format("%13.2f%%", mape));
        System.out.println("Max Absolute Error:                 $" + String.format("%,12.2f", maxAbsoluteError));
        System.out.println();
    }
    
    /**
     * Save Neural Network predictions to CSV
     */
    private static void saveNeuralNetworkToCSV(TwoLayerNN nn, DataSet data, String outputFile) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("Example,Actual_Price,Predicted_Price,Error,Percent_Error,Squared_Error");
            
            int count = 1;
            for (Example example : data.getData()) {
                double actualNormalized = example.getLabel();
                double predictedNormalized = nn.predictValue(example);
                
                double actualPrice = denormalize(actualNormalized);
                double predictedPrice = denormalize(predictedNormalized);
                double error = actualPrice - predictedPrice;
                double percentError = Math.abs(error / actualPrice * 100);
                double squaredError = error * error;
                
                writer.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f%n", 
                    count, actualPrice, predictedPrice, error, percentError, squaredError);
                count++;
            }
            
            System.out.println("Predictions saved to: " + outputFile);
            
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }
    }
    
    /**
     * Test KNN Classifier (adapted for regression)
     */
    private static void testKNN(DataSet trainData, DataSet testData) {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║       K-NEAREST NEIGHBORS (KNN)        ║");
        System.out.println("╚════════════════════════════════════════╝");
        
        // Test multiple K values
        int[] kValues = {1, 3, 5, 7};
        double bestMAE = Double.MAX_VALUE;
        int bestK = 3;
        
        for (int k : kValues) {
            KNNRegressor knn = new KNNRegressor();
            knn.setK(k);
            knn.train(trainData);
            
            double mae = calculateMAE(knn, testData);
            System.out.println("K=" + k + " -> MAE: $" + String.format("%.2f", mae));
            
            if (mae < bestMAE) {
                bestMAE = mae;
                bestK = k;
            }
        }
        
        System.out.println("\nBest K=" + bestK + "\n");
        
        KNNRegressor knn = new KNNRegressor();
        knn.setK(bestK);
        
        long startTime = System.currentTimeMillis();
        knn.train(trainData);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Training time: " + trainTime + " ms\n");
        
        generatePredictionsTable(knn, testData, "KNN (K=" + bestK + ")");
        savePredictionsToCSV(knn, testData, OUTPUT_DIR + "predictions_knn.csv");
    }
    
    /**
     * Test Decision Tree (adapted for regression)
     */
    private static void testDecisionTree(DataSet trainData, DataSet testData) {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║          DECISION TREE                 ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println("Note: Decision trees work best with discrete features\n");
        
        DecisionTreeRegressor dt = new DecisionTreeRegressor();
        dt.setDepthLimit(-1); // No depth limit
        
        long startTime = System.currentTimeMillis();
        dt.train(trainData);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Training time: " + trainTime + " ms\n");
        
        generatePredictionsTable(dt, testData, "Decision Tree");
        savePredictionsToCSV(dt, testData, OUTPUT_DIR + "predictions_dt.csv");
    }
    
    /**
     * Test Perceptron (adapted for regression)
     */
    private static void testPerceptron(DataSet trainData, DataSet testData) {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║            PERCEPTRON                  ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println("Note: Perceptron is linear, may not capture complex patterns\n");
        
        PerceptronRegressor p = new PerceptronRegressor();
        p.setIterations(100);
        
        long startTime = System.currentTimeMillis();
        p.train(trainData);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.println("Training time: " + trainTime + " ms\n");
        
        generatePredictionsTable(p, testData, "Perceptron");
        savePredictionsToCSV(p, testData, OUTPUT_DIR + "predictions_perceptron.csv");
    }
    
    /**
     * Calculate MAE for quick comparison
     */
    private static double calculateMAE(RegressorInterface regressor, DataSet data) {
        double totalAbsoluteError = 0.0;
        
        for (Example example : data.getData()) {
            double actualNormalized = example.getLabel();
            double predictedNormalized = regressor.predictValue(example);
            
            double actualPrice = denormalize(actualNormalized);
            double predictedPrice = denormalize(predictedNormalized);
            
            totalAbsoluteError += Math.abs(actualPrice - predictedPrice);
        }
        
        return totalAbsoluteError / data.getData().size();
    }
    
    /**
     * Generate and print predictions table
     */
    private static void generatePredictionsTable(RegressorInterface regressor, DataSet data, String modelName) {
        // Print header
        System.out.println("│ Example  │ Actual Price  │ Predicted Price │   Error    │ Percent Error │ Squared Error  │");
        
        // Track aggregate metrics
        double totalSquaredError = 0.0;
        double totalAbsoluteError = 0.0;
        double totalPercentError = 0.0;
        double maxAbsoluteError = 0.0;
        
        int count = 1;
        for (Example example : data.getData()) {
            double actualNormalized = example.getLabel();
            double predictedNormalized = regressor.predictValue(example);
            
            double actualPrice = denormalize(actualNormalized);
            double predictedPrice = denormalize(predictedNormalized);
            double error = actualPrice - predictedPrice;
            double percentError = Math.abs(error / actualPrice * 100);
            double squaredError = error * error;
            
            System.out.printf("│ %8d │ $%11.2f │ $%14.2f │ $%9.2f │ %12.2f%% │ $%13.2f │%n", 
                count, actualPrice, predictedPrice, error, percentError, squaredError);
            
            // Accumulate metrics
            totalSquaredError += squaredError;
            totalAbsoluteError += Math.abs(error);
            totalPercentError += percentError;
            maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
            
            count++;
        }
        
        // Calculate and display aggregate metrics
        int n = data.getData().size();
        double mse = totalSquaredError / n;
        double rmse = Math.sqrt(mse);
        double mae = totalAbsoluteError / n;
        double mape = totalPercentError / n;
        
        System.out.println("\n=== Performance Metrics (" + modelName + ") ===");
        System.out.println("Mean Squared Error (MSE):           $" + String.format("%,12.2f", mse));
        System.out.println("Root Mean Squared Error (RMSE):     $" + String.format("%,12.2f", rmse));
        System.out.println("Mean Absolute Error (MAE):          $" + String.format("%,12.2f", mae));
        System.out.println("Mean Absolute Percent Error (MAPE): " + String.format("%13.2f%%", mape));
        System.out.println("Max Absolute Error:                 $" + String.format("%,12.2f", maxAbsoluteError));
        System.out.println();
    }
    
    /**
     * Save predictions to CSV file
     */
    private static void savePredictionsToCSV(RegressorInterface regressor, DataSet data, String outputFile) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("Example,Actual_Price,Predicted_Price,Error,Percent_Error,Squared_Error");
            
            int count = 1;
            for (Example example : data.getData()) {
                double actualNormalized = example.getLabel();
                double predictedNormalized = regressor.predictValue(example);
                
                double actualPrice = denormalize(actualNormalized);
                double predictedPrice = denormalize(predictedNormalized);
                double error = actualPrice - predictedPrice;
                double percentError = Math.abs(error / actualPrice * 100);
                double squaredError = error * error;
                
                writer.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f%n", 
                    count, actualPrice, predictedPrice, error, percentError, squaredError);
                count++;
            }
            
            System.out.println("Predictions saved to: " + outputFile);
            
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }
    }
    
    /**
     * Denormalize a value from [0,1] to actual price
     */
    private static double denormalize(double normalized) {
        double clamped = Math.max(0.0, Math.min(1.0, normalized));
        return clamped * (MAX_PRICE - MIN_PRICE) + MIN_PRICE;
    }
    
    /**
     * Interface for regression models
     */
    interface RegressorInterface {
        void train(DataSet data);
        double predictValue(Example example);
    }
    
    /**
     * KNN adapted for regression (averages K nearest neighbors' prices)
     */
    static class KNNRegressor extends KNNClassifier implements RegressorInterface {
        @Override
        public double predictValue(Example example) {
            // Find K nearest neighbors
            List<Example> neighbors = findKNearestNeighbors(example, super.k);
            
            // Average their labels (normalized prices)
            double sum = 0.0;
            for (Example neighbor : neighbors) {
                sum += neighbor.getLabel();
            }
            
            return neighbors.isEmpty() ? 0.0 : sum / neighbors.size();
        }
        
        private List<Example> findKNearestNeighbors(Example example, int k) {
            List<DistanceExample> distances = new ArrayList<>();
            
            for (Example neighbor : super.examples) {
                if (neighbor == example) continue;
                double distance = euclideanDistance(example, neighbor);
                distances.add(new DistanceExample(neighbor, distance));
            }
            
            distances.sort((a, b) -> Double.compare(a.distance, b.distance));
            
            List<Example> result = new ArrayList<>();
            for (int i = 0; i < Math.min(k, distances.size()); i++) {
                result.add(distances.get(i).example);
            }
            
            return result;
        }
        
        private static class DistanceExample {
            Example example;
            double distance;
            
            DistanceExample(Example example, double distance) {
                this.example = example;
                this.distance = distance;
            }
        }
    }
    
    /**
     * Decision Tree adapted for regression (averages leaf values)
     */
    static class DecisionTreeRegressor extends DecisionTreeClassifier implements RegressorInterface {
        @Override
        public double predictValue(Example example) {
            // Decision trees predict discrete values, so we just return the classification
            // This is a limitation - true regression trees would average leaf values
            return classify(example);
        }
    }
    
    /**
     * Perceptron adapted for regression (linear model)
     */
    static class PerceptronRegressor extends PerceptronClassifier implements RegressorInterface {
        @Override
        public double predictValue(Example example) {
            // Use the weighted sum directly as prediction (linear regression)
            double sum = 0.0;
            for (Integer feature : example.getFeatureSet()) {
                sum += super.weights.get(feature) * example.getFeature(feature);
            }
            return sum; // Return raw sum, not thresholded
        }
        
        @Override
        public void train(DataSet data) {
            // Initialize weights
            for (Integer feature : data.getAllFeatureIndices()) {
                super.weights.put(feature, 0.0);
            }
            
            // Simple gradient descent for regression
            double learningRate = 0.01;
            
            for (int iter = 0; iter < 100; iter++) {
                for (Example example : data.getData()) {
                    double prediction = predictValue(example);
                    double actual = example.getLabel();
                    double error = actual - prediction;
                    
                    // Update weights
                    for (Integer feature : example.getFeatureSet()) {
                        double oldWeight = super.weights.get(feature);
                        double newWeight = oldWeight + learningRate * error * example.getFeature(feature);
                        super.weights.put(feature, newWeight);
                    }
                }
            }
        }
    }
}