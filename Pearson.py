import numpy as np

def feed_training_data(input):
    i=0
    file = open('train.txt', 'r')
    for line in file.readlines():
        input[i] = [int(x) for x in line.split()]
        i=i+1

def test_pearson_similarity(input,file,num,k):
    testdata = open(file, 'r').read().strip().split('\n')
    testmatrix = [data.split() for data in testdata]
    testmatrix = [[int(e) for e in data] for data in testmatrix]
    current_user_id = testmatrix[0][0]-1
    current_user_data = [0]*1000
    movie_ids = []
    ratings=[]
    outputfile = "result"+str(num)+".txt"

    #Reading test file and looping every row
    for i in range(len(testmatrix)):
        user_id = testmatrix[i][0]-1
        movie_id = testmatrix[i][1]-1
        rating = testmatrix[i][2]

        if user_id== current_user_id:
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating
        else:
            get_results_using_pearson(
                input,
                current_user_data,
                current_user_id,
                movie_ids,
                ratings,
                outputfile,
                k,
                None,
                True
            )
            current_user_id = user_id
            current_user_data = [0] * 1000
            movie_ids = []
            if rating == 0 :
                movie_ids.append(movie_id)
            else:
                current_user_data [movie_id] = rating

    get_results_using_pearson(input,current_user_data,current_user_id,movie_ids,ratings,outputfile,k,None,True)
    return ratings

def get_results_using_pearson(input,current_user_data,current_user_id,movie_ids,ratings,outputfile,k,caseAmp=None,isIUF=None):
     weights=[]
     avg_ratings=[0]*200
     current_user_avg_rating = 0

     #Caluclating average rating for each user
     for i,row in enumerate(input):
         Sum=sum(i for i in row)
         count=len([i for i in row if i > 0])
         if count>0:
             avg_ratings[i]=Sum/count

     #Calulating average rating for current user
     Sum=sum(i for i in current_user_data)
     count=len([i for i in current_user_data if i > 0])
     current_user_avg_rating=Sum/count


     IUFratings=np.copy(input)

     #If IUF multiplying input matrix with IUF factor
     if isIUF is not None:
         for i in range(1000):
             count=0
             for row in input:
                if row[i] > 0:
                    count=count+1
             if count == 0:
                 continue
             iuf=np.log(200/count)
             current_user_data[i] = np.rint(current_user_data[i] *iuf)
             for u in IUFratings:
                 u[i] = np.rint(u[i] * iuf)

     if isIUF is not None:
         for i,row in enumerate(IUFratings):
             weights.append(get_pearson_similarity(row,current_user_data,avg_ratings[i],current_user_avg_rating))
     else:
         for i,row in enumerate(input):
            weights.append(get_pearson_similarity(row,current_user_data,avg_ratings[i],current_user_avg_rating))

     # If case amplification calculating new weights
     if caseAmp is not None:
         weights = [w * np.abs(w)**(caseAmp-1) for w in weights]

     # Creating output file
     file = open(outputfile, "a")

     for movie in movie_ids:
         wsum=0
         weight_sum=0
         rating =0
         results = []
         for i, val in enumerate(weights):
             if input[i][movie] == 0:
                 continue
             results.append((i, val))

         # Finding k nearest neighbors
         results.sort(key=sortSecond, reverse=True)
         del results[k:]

         for row in results:
             rating=0
             user_id = row[0]
             sim = row[1]

             if input[user_id][movie]==0:
                 continue

             weight_sum+=abs(sim)
             new_rating = input[user_id][movie] - avg_ratings[user_id]
             wsum+=sim*new_rating

         if weight_sum != 0:
             rating = wsum/weight_sum
         else:
         # If no neighbours were found, guess a score of 3.
             rating = 3

         rating += current_user_avg_rating
         rating = scale_rating(rating)

         # Appending results to output file
         file.write("%i %i %i\n" % (current_user_id+1, movie+1,rating))
         ratings.append(rating)

     file.close()
     return

def scale_rating(rating):
    rating = int(np.rint(rating))
    if rating > 5:
        rating = 5
    elif rating < 1:
        rating = 1

    return rating

def get_common_values(v1, v2):
    """
    Returns new vectors (a, b) where both vectors have values for given index
    """
    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y > 0 and x > 0:
            v1_new.append(x)
            v2_new.append(y)

    return np.array(v1_new), np.array(v2_new)

def get_pearson_similarity(user1,user2,user1_avg,user2_avg):
    """
        Cosine similarity between two vectors.
        Value ranges from [-1, 1].
    """

    vector_v1, vector_v2 = get_common_values(user1, user2)
    vector_v1 = [x - user1_avg for x in vector_v1]
    vector_v2 = [y - user2_avg for y in vector_v2]

    sim = np.dot(vector_v1, vector_v2)

    lenght_v1 = get_length(vector_v1)
    length_v2 = get_length(vector_v2)


    if lenght_v1 != 0 and length_v2 != 0:
        sim /= (lenght_v1 * length_v2)
    else:
        sim = 0

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim

def get_length(v):
    return np.sqrt(np.sum(np.square(v)))

def sortSecond(val):
    return val[1]

def main():
    # Declaring rows
    N = 200
    # Declaring columns
    M = 1000

    #Intializing input matrix
    input = [[0 for i in range(M)] for j in range(N)]
    feed_training_data(input)

    print("processing test file 5")
    test_pearson_similarity(input,"test5.txt",5,50)
    print("predictions done for file 5")
    print("processing test file 10")
    test_pearson_similarity(input,"test10.txt",10,50)
    print("predictions done for file 10")
    print("processing test file 20")
    test_pearson_similarity(input, "test20.txt",20,50)
    print("predictions done for file 20")

main()