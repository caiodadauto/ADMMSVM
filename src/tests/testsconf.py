tests_conf = {
        'Test for the Secom data set (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'secom',
            'data_info': {
                'file': ['src/tests/traindatas/secom/secom.data', 'src/tests/traindatas/secom/secom_labels.data'],
                'delimiter': ' ',
                'class_split': True,
                'header': None
                }
            },
        'Test for the credit card risk (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'credit card',
            'data_info': {
                'file': ['src/tests/traindatas/creditcard/credit_card.csv'],
                'delimiter': ',',
                'class_split': False,
                'index': 0,
                'label':{
                    -1: 0
                    }
                }
            },
        'Test for the artificial data set (linear).' : {
            'get_nodes': True,
            'type': 'linear',
            'name': 'artificial',
            'data_info': {
                'file': 'artificial linear',
                'delimiter': None,
                'class_split': None,
                'auto_gen': True
                }
            },
        'Test for the pima indians diabetes data set (nonlinear).' : {
            'get_nodes': False,
            'type': 'nonlinear',
            'name': 'diabetes',
            'data_info': {
                'file': ['src/tests/traindatas/diabetic/pima-indians-diabetes.data'],
                'delimiter': ',',
                'class_split': False,
                'header': None,
                'label':{
                    -1: 0
                    },
                'encode_null': 0
                }
            },
        'Test for the artificial circles data set (nonlinear).' : {
            'get_nodes': False,
            'type': 'nonlinear',
            'name': 'circles',
            'data_info': {
                'file': 'circles',
                'delimiter': None,
                'class_split': None,
                'auto_gen': True
                }
            }
        }
