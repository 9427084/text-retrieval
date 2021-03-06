package at.ac.tuwien.ifs.ir;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import at.ac.tuwien.ifs.ir.TextRetrieval.postingListSize;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CosineSimilarity {

    private static Logger log = LoggerFactory.getLogger(CosineSimilarity.class);
    
    private String weightsFile = null;
    private String target = ".";
    private postingListSize postingListSize;
    
    public CosineSimilarity(String weightsFile, String target, TextRetrieval.postingListSize postingListSize) {
        this.weightsFile = weightsFile;
        this.target = target;
        this.postingListSize = postingListSize; 
    }
    
    public void findSimilar(String[] documentIDs) {
        log.info("Started similarity retrieval ...");
        for (int i = 0; i < documentIDs.length; i++)
            findSimilar(documentIDs[i], i + 1);
        log.info("Done similarity retrieval");
    }
    
    public void findSimilar(String documentID, int topicNumber) {
        log.info("Started searching for similar documents for " + documentID + " ...");
        
        Instances weights = null;
        try {
            weights = new DataSource(weightsFile).getDataSet();

            int queryIndex;
            boolean found = false;
            for (queryIndex = 0; queryIndex < weights.numInstances(); queryIndex++) {
                if ((weights.instance(queryIndex).stringValue(0)).equals(documentID)) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                log.warn("Requested query document with documentID of " + documentID + " could not be found");
                return;
            }
            
            Rank[] top10Ranks = new Rank[10];
            // initialize with something smaller than 0
            // the cosine similarity here is between 0 and 1, because the weights are not negative
            // therefore the angle between the vectors will not be greater than 90° meaning a 0
            // generally cosine similarity will vary between -1 and 1
            for (int i = 0; i < 10; i++)
                top10Ranks[i] = new Rank(-1.1, null);

            for (int i = 0; i < weights.numInstances(); i++) {
                if (i == queryIndex)
                    continue;

                double newSimilarity = computeCosineSimilarity(weights, queryIndex, i);
                
                if (top10Ranks[0].similarity < newSimilarity)
                    top10Ranks[0] = new Rank(newSimilarity, weights.instance(i).stringValue(0));
                Arrays.sort(top10Ranks);
            }
            log.info("Found 10 most similar documents for " + documentID);

            String filename = target + "/" + postingListSize + "_topic" + topicNumber + "_groupG.txt";
            try {
                File file = new File(filename.substring(0, filename.lastIndexOf("/")));
                if (!file.exists()) { 
                    file.mkdirs();
                }
                file = new File(filename);
                if (!file.exists()) { 
                    file.createNewFile();
                }
                
                FileWriter fileWriter = new FileWriter(filename);
                BufferedWriter out = new BufferedWriter(fileWriter);
                Collections.reverse(Arrays.asList(top10Ranks));
                for (int i = 0; i < 10; i++) {
                    out.write("topic" + topicNumber
                            + " Q0 " + top10Ranks[i].documentID 
                            + " " + (i + 1)
                            + " " + top10Ranks[i].similarity
                            + " groupG_" + postingListSize + "\r\n");
                }
                out.close();
                
                log.info("Wrote results to " + filename);
            } catch (IOException ioe) {
                log.error("Error saving results to file " + filename, ioe);
                return;
            }
            log.info("Done");

        } catch (Exception e) {
            log.error("Error computing cosine similarities", e);
            return;
        }
    }

    private double computeCosineSimilarity(Instances weights, int queryIndex, int documentIndex) {

        double dotProduct = 0.0;
        double queryEuclidean = 0.0;
        double documentEuclidean = 0.0;
        
        for (int i = 2; i < weights.numAttributes(); i++ ) {
            
            double queryValue = weights.instance(queryIndex).value(i);
            queryValue = Double.isNaN(queryValue) ? 0 : queryValue;
            double documentValue = weights.instance(documentIndex).value(i);
            documentValue = Double.isNaN(documentValue) ? 0 : documentValue;
            
            dotProduct += queryValue*documentValue;
            queryEuclidean += Math.pow(queryValue, 2);
            documentEuclidean += Math.pow(documentValue, 2);
        }
        double euclideans = Math.sqrt(queryEuclidean) * Math.sqrt(documentEuclidean);
        return euclideans == 0.0 ? 0.0 : dotProduct / euclideans;
    }

    private class Rank implements Comparable<Rank> {

        public Double similarity;
        public String documentID;
        public Rank(Double similarity, String documentID) {
            this.similarity = similarity;
            this.documentID = documentID;
        }

        @Override
        public int compareTo(Rank anotherRank) {
            if (anotherRank == null)
                return 1;
            return similarity.compareTo(anotherRank.similarity);
        }
    }
}
