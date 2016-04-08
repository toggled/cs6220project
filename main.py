from src.Lda import Lda


__author__ = 'Naheed'

topic_bins = []
load_pickled = True  # Whether want to load pickled bins from pk file

def run_phaseone():
    if load_pickled:
        import pickle
        with open('./test/topicbins.pk') as f:
            topic_bins = pickle.load(f)

    else:
        ld = Lda()
        ld.runlda()
        topic_bins = ld.bin  # List of List(product_id)

    for i in topic_bins:
        if i:
            print len(i)

def run_phasetwo():
    '''
    TO DO:
     1. Extract Features from training set, product description and attributes of the docs belonging to each bin.
     2. Train K models on them.
     3.
    '''
    pass


def get_prediction():
    '''
    Function which will predict the rank for test query.
    TO DO:
     1. Get the product title and compute similarity with the topics.
     2. assign normalized similarity as weights to each to topics./
     3. Compute features based on Product title,search term
     4. Get Rank from K Models and compute weighted average
    '''
    pass


if __name__ == '__main__':
    run_phaseone()
