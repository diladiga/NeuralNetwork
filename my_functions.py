import glob
import xarray as xr
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def read_netcdfs(files, dim):
    ''' 
    Function to combine several files that have the same
    variables into one single xarray.Dataset
    
    files: all the files to combine
    dim: the dimension used to combine the files
    returns: xarray.Dataset
    '''

    paths = sorted(glob.glob(files), key=alphanum_key)
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)

    return combined

def build_nn_eq(nds_num,inpt_num,inpt_list,act_func):
    '''
    Function to build the NN equation from the weight and
    biases outputs
    '''

    # temporal list as a container for each layer formulas
    formula_list = []
    frml_eqn = []
    y_str = []
    for ii in np.arange(len(model.layers)):
         # get ith Keras layer's weights and biases
         layer = model.layers[ii]
         WB = layer.get_weights()

         #WB[0].shape = (2,2)
         # empty text string to which concatenate current layer formula parts
         formula = ''
         if ii==0:

             for jj in np.arange(nds_num):
                weights = []
                all_terms = [ ]

                for kk in np.arange(inpt_num):
                    cur_weight = WB[0][kk][jj]
                    cur_bias = WB[1][jj]
                    weights.append(cur_weight)
                    # build formula for this layer
                    term = (str(np.round(weights[kk],2))+inpt_list[kk]+'+' )
                    all_terms.append(term)

                bias = str(np.round(cur_bias,2))
                all_terms.append(bias)

                formula_list.append(all_terms)



        elif ii == (len(model.layers)-1):
            for ll in np.arange(nds_num):
                act_term = ''
                for item in formula_list[ll]:
                    act_term += str(item)
                y_str.append( str(np.round(WB[0][ll],2).squeeze()) + '('+ act_func+ '(' + act_term +'))+' )

            y_str.append(str(np.round(WB[1][0],2).squeeze()) )
 equation = ' '
    for item in y_str:
        equation += str(item)

    # make some cleanings
    equation = equation.replace('+-','-')
    equation = equation.replace('+*0.0*','')
    equation = equation.replace('-*0.0*','')
    equation = equation.replace('*','')

    return equation

## min max noralization
def MinMaxNorm(x,a,b):
    ''' Min-max normalization method between a specified range
        Normalization range = [a,b]
        x = input data
        a = lower range
        b = upper range
        This method normalizes the input and output variables for the training of the NN'''

    n = (b-a)*(x-np.min(x))/(np.max(x)-np.min(x))+a
    return n

def MinMaxInverse(n,a,b,df_tsr):
    ''' Min-max inverse transform method
        n = normalized data between [a,b]
        a = lower range
        b = upper range

        This method transforms the range of the predicted values using the NN,
        from -1 to 1 to the correct range of the magnitude of the fnt of sw at TOA'''

    #we define the min and max values from the training dataset
    max_val = np.round(np.max(df_tsr),2)
    min_val = np.round(np.min(df_tsr),2)

    x = (n-a)*(max_val-min_val)/(b-a)+min_val
    return x

#####  Function to train the NN (Heuristic case)
def runNN_h(x_values,y_real,alpha, iterations,shape_input,nodesLayer1, nodesLayer2, act_function1, act_function2, patienceEpochs ,dpii=80):

    start_time = time.time()

    # defining the tensors for the NN
    x=tf.constant(x_values) #albedo
    y=tf.constant(y_real) # prtrbd.fnt_sw_toa.to_numpy()

    # define the model
    model = tf.keras.Sequential(name='Sequential_NN')

    if nodesLayer2 == 0:
        layer1 = Dense(nodesLayer1,activation=act_function1 ,input_shape=[shape_input], name='hiddenLayer1') #11 relu
        output = Dense(1, name='output')

        model.add(layer1)
        model.add(output)

    else:
        layer1 = Dense(nodesLayer1,activation=act_function1 ,input_shape=[shape_input], name='hiddenLayer1') #11 relu
        layer2 = Dense(nodesLayer2,activation=act_function2, name='hiddenLayer2')
        output = Dense(1, name='output')

        model.add(layer1)
        model.add(layer2)
        model.add(output)

    # compile the model
    model.compile(loss='mse',
                  optimizer= tf.keras.optimizers.Adam(learning_rate=alpha),
                  metrics=['accuracy'])

    # display the model
    model.summary()
    ann_viz(model, filename='figure_NN', title="Neural network")

    # save hyperparameters
    weights_dict = {}

    weight_callback = tf.keras.callbacks.LambdaCallback \
    ( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))

    #adding a callback to stop training
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=patienceEpochs)
 # fit the model
    history = model.fit( x, y, epochs=iterations, callbacks=[weight_callback,es ],verbose=True)
    #print(history.history)

    # get the learning rate value of the model
    lr= K.eval(model.optimizer.lr)

    # plot cost function
    plt.figure(figsize=(8,6),dpi=dpii)
    history_df = pd.DataFrame(history.history)

    plt.plot(history_df['loss'], label='cost')
    plt.yscale('log')

    plt.title('Training cost function with learning rate = '+ str(lr))
    plt.legend()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))

    return(model,history_df,weights_dict)

#####  Function to train the NN (Global case)
def runNN(x_values,y_real,alpha, iterations,shape_input,nodesLayer1, nodesLayer2, act_function1, act_function2, 
          earlyStopping, patienceEpochs, loss_thr ,dpii=300):
    
    start_time = time.time()
    
    # defining the tensors for the NN
    x=tf.constant(x_values) #albedo
    y=tf.constant(y_real) # prtrbd.fnt_sw_toa.to_numpy()
    
    # define the model
    model = tf.keras.Sequential(name='Sequential_NN')
    
    if nodesLayer2 == 0:
        layer1 = Dense(nodesLayer1,activation=act_function1 ,input_shape=[shape_input], name='hiddenLayer1') #11 relu
        output = Dense(1, name='output')

        model.add(layer1)
        model.add(output)
    
    else:
        layer1 = Dense(nodesLayer1,activation=act_function1 ,input_shape=[shape_input], name='hiddenLayer1') #11 relu
        layer2 = Dense(nodesLayer2,activation=act_function2, name='hiddenLayer2')
        output = Dense(1, name='output')

        model.add(layer1)
        model.add(layer2)
        model.add(output)

    # compile the model
    model.compile(loss='mse',
                  optimizer= tf.keras.optimizers.Adam(learning_rate=alpha), 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]) #'accuracy'

    # display the model
    model.summary()
    ann_viz(model, filename='figure_NN', title="Neural network")
    
    # save hyperparameters
    weights_dict = {}

    weight_callback = tf.keras.callbacks.LambdaCallback \
    ( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))
    
    if earlyStopping==0:
        # fit the model
        history = model.fit( x, y, epochs=iterations, callbacks=[weight_callback],verbose=True)
    
    if earlyStopping== 1:
        #adding a callback to stop training 
        #es = EarlyStoppingByLossVal(monitor='loss', value=loss_thr, verbose=True)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, mode='min', 
                                             patience=patienceEpochs)#min_delta=0.00001, start_from_epoch=10)
        # NEED TO CHECK THIS
        #es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, mode='min', 
                                              #baseline=loss_thr, min_delta=0.000001, patience=patienceEpochs)#min_delta=0.00001, start_from_epoch=10)

        # fit the model
        history = model.fit( x, y, epochs=iterations, callbacks=[weight_callback,es],verbose=2)
        #print(history.history)
    
    # get the learning rate value of the model
    lr= K.eval(model.optimizer.lr)

    # plot cost function
    plt.figure(figsize=(8,6),dpi=dpii)
    history_df = pd.DataFrame(history.history)
    plt.plot(history_df['loss'], label='cost')
    plt.yscale('log')
    plt.title('Training cost function with learning rate = '+ str(lr))
    plt.legend()
    plt.savefig('costFunction.png')
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return(model,history_df,weights_dict)
