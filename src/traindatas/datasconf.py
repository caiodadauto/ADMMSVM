datas_conf = {
        'Secom':{
            'file': ['src/traindatas/secom/secom.data', 'src/traindatas/secom/secom_labels.data'],
            'delimiter': ' ',
            'class_split': True,
            'header': None
            },
        'Credit Card':{
            'file': ['src/traindatas/creditcard/credit_card.csv'],
            'delimiter': ',',
            'class_split': False,
            'header': 'infer',
            'index': 0,
            'label':{
                -1: 0
                } 
            }
        }
