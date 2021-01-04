## Description:

**sentclassifier** is used to classify sentences of PubMed papers based on machine learing models.

## Usage:

```shell
    ./sentclassifier.sh <-i str> [-m str] [-n int]
```

## Parameters:

**-i**: input file with a header in following TSV format:  
    pmid<TAB>sentid<TAB>senttext<TAB>label

**-m**: machine learning model to use. choose from ['svm', 'rf', 'nb', 'knn']:  
    'svm': Support Vector Machine (Default)  
    'rf': Random Forest  
    'nb': Multinomial Naive Bayes  
    'knn': K-NearestNeighbor    

**-o**: directory name for saving trained model. default is previous directory.  

**-n**: number of threads to be used. default=1

## Example:

Use following command to run an example:

```shell
    ./sentclassifier.sh -i testdata/test.tsv -m svm -n 1
```

