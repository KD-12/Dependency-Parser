# Dependency-Parser
Type of Files and Command to run them
1]Basic Neural Net
    DependencyParser.py
    run - python DependencyParser.py
    contains :
             Loss Function DependencyParserModel.build_graph(...)
             This function also has the call to optimiser with non clipped gradients. This is commented.
             Forward pass with cubic Activation function : forward_pass(...)
             Also this file contains all other activation functions which are commented but were run and their results were
             recorded in the report
             Feature Generation :getFeatures(...)
             Apply Function in ParserSystem.py
2]Neural net with two hidden Layers:
    DependencyParser_hidden_layer_2.py
    run - python DependencyParser_hidden_layer_2.py
    contains :
              Loss Function DependencyParserModel.build_graph(...)
              this function calls forward_pass 2 times for finding predictions to compute loss.
              Forward pass with tanh Activation function : forward_pass(...)
              Also this file contains all other activation functions which are commented but were run and their results were
              recorded in the report
              Feature Generation :getFeatures(...)
              Apply Function in ParserSystem.py
3]Neural net with three hidden Layers: ################## (My Best Model) ###################
    DependencyParser_hidden_layer_3.py
    run - python DependencyParser_hidden_layer_3.py
    contains :
              Loss Function DependencyParserModel.build_graph(...)
              This function also has the call to optimiser with non clipped gradients. This is commented.
              this function calls forward_pass 3 times for finding predictions to compute loss.
              Forward pass with tanh Activation function : forward_pass(...)
              Also this file contains all other activation functions which are commented but were run and their results were
              recorded in the report
              Feature Generation :getFeatures(...)
              Apply Function in ParserSystem.py

4]Neural Net with three parallel hidden layers for POS, Labels and Tags:
    DependencyParser_parallel.py
    run python DependencyParser_parallel.py
    contains:
            Loss Function DependencyParserModel.build_graph(...)
            this function calls forward_pass 3 times once for each parallel hidden layer for finding predictions to compute loss.
            Forward pass with cubic Activation function : forward_pass(...)
            Also this file contains all other activation functions which are commented but were run and their results were
            recorded in the report
            Feature Generation :getFeatures(...)
            Apply Function in ParserSystem.py

5]Neural Net without training:
    DependencyParser_fixed.py
    run python DependencyParser_fixed.py
    contains:
            Loss Function DependencyParserModel.build_graph(...)

            Forward pass with cubic Activation function : forward_pass(...)
            Also this file contains all other activation functions which are commented but were run and their results were
            recorded in the report
            Feature Generation :getFeatures(...)
            Apply Function in ParserSystem.py
6]Config.py
Contains all configurations for hyper parameters.

