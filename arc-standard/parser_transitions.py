class PartialParse(object):
    def __init__(self, sentence):
        """
        Initializes this partial parse.

        Inputs:
            - sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not alter it in your code.
        self.sentence = sentence

        # YOUR CODE HERE (3 Lines)
        # Your code should initialize the following fields:
        # self.stack: The current stack represented as a list with the top of the stack as the
        # last element of the list.
        # self.buffer: The current buffer represented as a list with the first item on the
        # buffer as the first item of the list
        # self.dependencies: The list of dependencies produced so far. Represented as a list of
        # tuples where each tuple is of the form (head, dependent).
        # Order for this list doesn't matter.
        ###
        # Note: The root token should be represented with the string "ROOT"
        ###

        self.stack = ["ROOT"]
        self.buffer = sentence
        self.dependencies = []
        # END YOUR CODE


    def parse_step(self, transition):
        """
        Performs a single parse step by applying the given transition to this partial parse

        Inputs:
            - transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """
        # YOUR CODE HERE (~7-10 Lines)
        # TODO:
        # Implement a single parsing step, i.e. the logic for the following
        # 1. Shift
        # 2. Left Arc
        # 3. Right Arc

        if transition == "S":
            self.stack.append(self.buffer.pop(0))
        elif transition.startswith("L"):
            self.dependencies.append((self.stack[-1], self.stack.pop(-2), transition[2:]))
        elif transition.startswith("R"):
            self.dependencies.append((self.stack[-2], self.stack.pop(-1),transition[2:]))
        else:
            raise ValueError(f"Unknown transition: {transition}")
        # END YOUR CODE

    def parse(self, transitions):
        """
        Applies the provided transitions to this PartialParse

        Inputs:
            - transitions (list of str): The list of transitions in the order they should be applied
        Outputs:
            - dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    # sourcery skip: inline-immediately-returned-variable
    """
    Parses a list of sentences in minibatches using a model.

    Inputs:
        - sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
        - model (ParsingModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
        - The number of PartialParses to include in each minibatch
    Outputs:
        - dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []
    # YOUR CODE HERE (~8-10 Lines)
    # TODO:
    # Implement the minibatch parse algorithm, which will speedup parsing.
    ###
    # Note: A shallow copy can be made with the "=" sign in python, e.g.
    # unfinished_parses = partial_parses[:].
    # Here `unfinished_parses` is a shallow copy of `partial_parses`.
    # In Python, a shallow copied list like `unfinished_parses` does not contain new instances
    # of the object stored in `partial_parses`. Rather both lists refer to the same objects.
    # In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
    # contains references to the same objects. Thus, you should NOT use the `del` operator
    # to remove objects from the `unfinished_parses` list. This will free the underlying memory that
    # is being accessed by `partial_parses` and may cause your code to crash.

    partial_parses = [PartialParse(sentence) for sentence in sentences]

    unfinished_parses = partial_parses[:]

    while len(unfinished_parses) is not 0:
        mini_batch = unfinished_parses[:batch_size]
        transitions = model.predict(mini_batch)
        for transition, partial_parse in zip(transitions, mini_batch):
            partial_parse.parse_step(transition)
            if len(partial_parse.buffer) is 0 and len(partial_parse.stack) is 1:
                unfinished_parses.remove(partial_parse)
            else:
                pass

    dependencies = [partial_parse.dependencies for partial_parse in partial_parses]

    # END YOUR CODE

    return dependencies
