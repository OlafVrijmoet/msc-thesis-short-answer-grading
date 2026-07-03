
def read_xml(root, column_data, df_info, domain):

    # no max points should end up None in max_points column!
    max_points_input = None

    # saves question
    question_text = ""

    # save / create question ID - Still do!
    question_id = None

    # save reference answer
    reference_answer_texts = []

    # place to save student answers and points
    student_answer_texts = []
    assigned_points_list = []

    # loop through each student answer
    for nodes in root:

        # save the question
        if nodes.tag == "questionText":
            question_text = nodes.text

        # loop through nodes
        for node in nodes:

            # add reference answers
            if node.tag == "referenceAnswer":

                # if good encountered --> 2 poinst possible - GOOD not necissarily worth 1 of 2 points!
                # if node.attrib["category"] == "GOOD":
                #     max_points_input = 2

                    # add it to student answers!

                # if max_points_input is None no good encountered yet so max_poinst = 1
                if max_points_input == None:
                    max_points_input = 1

                if "category" in node.attrib:
                    # add every "BEST" reference answer
                    if node.attrib["category"] == "BEST":

                        # add every "BEST" reference answer
                        reference_answer_texts.append(node.text)
                
                # check if this is the only reference answer, than add it to ass correct answer
                elif len(nodes) == 1:
                    # add reference answer text to reference answers
                    reference_answer_texts.append(node.text)
                else:
                    print("\n-------!!!!!!!!--------")
                    print("More than 1 reference answer and no category attribute")
                    print("-------!!!!!!!!--------\n")
            
            # add student answers
            if node.tag == "studentAnswer":
                
                # save answer
                student_answer_text = node.text

                # add student answer ID
                student_answer_id = node.attrib["id"]

                student_answer_texts.append(student_answer_text)

                # give points
                if node.attrib["accuracy"] == "correct":

                    assigned_points_list.append(max_points_input)

                elif node.attrib["accuracy"] == "incorrect":

                    assigned_points_list.append(0)
                else:
                    # add null if answer is neither correct or incorrect
                    assigned_points_list.append(None)
                    
        if len(student_answer_texts) is not len(assigned_points_list):
            print("\n-------!!!!!!!!--------")
            print("student_answer_texts and assigned_points_list not the same length!")
            print("-------!!!!!!!!--------\n")
        
    # add answers to lists for df
    # add all student answers to every correct reference answer
    for reference_a in reference_answer_texts:

        # create unique question id
        former_question = ""

        # add every student answer
        for index, student_a in enumerate(student_answer_texts):
            
            if former_question != question_text:
                df_info.question_id_count += 1

            # add new row in df
            column_data.row_id.append(df_info.row_id_count)
            column_data.question.append(question_text)
            column_data.question_id.append(df_info.question_id_count)
            column_data.student_answer.append(student_a)
            column_data.reference_answer.append(reference_a)
            column_data.assigned_points.append(assigned_points_list[index])
            column_data.max_points.append(max_points_input)
            column_data.domain.append(domain)

            df_info.row_id_count += 1
            former_question = question_text
