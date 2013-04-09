package at.ac.tuwien.ifs.ir;

import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class TFIDF {

    private static Logger log = LoggerFactory.getLogger(TFIDF.class);

    private String indexFile = null;
    private String weightsFile = null;

    public TFIDF(String indexFile, String weightsFile) {
        this.indexFile = indexFile;
        this.weightsFile = weightsFile;
    }

    public void createWeights() {
        log.info("Started creating TF-IDF weights ...");
        
        Instances index = null;
        try {
            index = new DataSource(indexFile).getDataSet();
            
            log.info("Processing " + index.numAttributes() + " terms");

            for (int i = 2; i < index.numAttributes(); i++) {
                if (i % 500 == 0)
                    log.info(i + " terms done ...");
                Attribute attribute = index.attribute(i);
                
                // compute the IDF for the current term
                // we use 1 + |{d;t e d}|, term t, document d for the DF to avoid division by 0
                int containingTerm = 1;
                for (int j = 0; j < index.numInstances(); j++) {
                    if (index.instance(j).value(attribute) > 0)
                        containingTerm++;
                }
                double idf = Math.log10((double) index.numInstances() / (double) containingTerm);
                
                // compute the TF and the TF-IDF
                for (int j = 0; j < index.numInstances(); j++) {
                    Instance currentInstance = index.instance(j);
                    currentInstance.setValue(attribute, Math.log(1 + currentInstance.value(attribute)) * idf);
                }
            }
            log.info(index.numAttributes() + " terms done");
        } catch (Exception e) {
            log.error("Error creating TF-IDF values", e);
            return;
        }
        
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(index);
            saver.setFile(new File(weightsFile));
            saver.setCompressOutput(true);
            saver.writeBatch();
            log.info("Wrote TF-IDF values to " + weightsFile);
        } catch (IOException ioe) {
            log.error("Error saving TF-IDF weights to ARFF file", ioe);
            return;
        }
        log.info("Done creating TF-IDF weights");
    }
}
