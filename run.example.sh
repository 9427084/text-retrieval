java -jar target/TextRetrieval-1.0.0-SNAPSHOT.jar index --source 20_newsgroups_subset --target 20_newsgroups_subset_RESULT --stemming true -b 500 --lower 3 --upper 19 
java -jar target/TextRetrieval-1.0.0-SNAPSHOT.jar weight --indexFile 20_newsgroups_subset_RESULT/index.arff.gz --weightsFile 20_newsgroups_subset_RESULT/weights.arff.gz
java -jar target/TextRetrieval-1.0.0-SNAPSHOT.jar compare --weightsFile 20_newsgroups_subset_RESULT/weights.arff.gz --target 20_newsgroups_subset_RESULT
