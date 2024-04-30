
def import_log_properties() -> dict:
    log_properties = {'BP12-AW': {'not-data-attributes': ['time:timestamp', 'case:concept:name',
                                                          'concept:name', 'case:REG-DATE', 'REG-DATE', 'lifecycle:transition'],
                                  'categorical-attributes': ['org:resource']},
                      'bpic12-a': {'not-data-attributes': ['time:timestamp', 'case:concept:name',
                                                         'concept:name', 'case:REG-DATE', 'REG-DATE', 'lifecycle:transition'],
                                  'categorical-attributes': ['org:resource']},
                      'BPIC2017-OfferLog': {'not-data-attributes': ['time:timestamp', 'case:concept:name', 
                                                          'concept:name', 'EventID', 'lifecycle:transition',
                                                          'case:ApplicationID', 'ApplicationID', 'OfferID', 'EventOrigin'],
                                   'categorical-attributes': ['org:resource', 'Action']},
                      'bpic2020-DomesticDeclarations': {'not-data-attributes': ['time:timestamp', 'case:concept:name',
                                                                               'concept:name', 'case:BudgetNumber', 'BudgetNumber',
                                                                               'case:DeclarationNumber', 'DeclarationNumber', 'case:id', 'id'],
                                                        'categorical-attributes': ['org:resource', 'org:role']},
                      'bpic2020-InternationalDeclarations': {'not-data-attributes': ['case:concept:name', 'case:id', 'id', 
                                                                                    'concept:name', 'time:timestamp'],
                                                             'categorical-attributes': ['org:resource', 'org:role',
                                                                                        'case:Permit travel permit number',
                                                                                        'case:DeclarationNumber', 'case:Permit TaskNumber',
                                                                                        'case:Permit BudgetNumber', 'case:Permit ProjectNumber', 
                                                                                        'case:Permit OrganizationalEntity', 'case:travel permit number',
                                                                                        'case:Permit ID', 'case:Permit id', 'case:BudgetNumber',
                                                                                        'case:Permit ActivityNumber']},
                      'bpic2020-PrepaidTravelCost': {'not-data-attributes': ['case:concept:name', 'case:id', 'id', 
                                                                             'concept:name', 'time:timestamp', 'case:Cost Type'],
                                                             'categorical-attributes': ['org:resource', 'org:role',
                                                                                        'case:Permit travel permit number', 'case:OrganizationalEntity',
                                                                                        'case:DeclarationNumber', 'case:Permit TaskNumber', 'case:Task',
                                                                                        'case:Permit BudgetNumber', 'case:Permit ProjectNumber', 'case:Project',
                                                                                        'case:Permit OrganizationalEntity', 'case:travel permit number', 'case:Activity',
                                                                                        'case:Permit ID', 'case:Permit id', 'case:BudgetNumber', 'case:RfpNumber', 
                                                                                        'case:Permit ActivityNumber', 'case:Rfp-id']},
                      'bpic2020-RequestForPayment': {'not-data-attributes': ['id', 'concept:name', 'case:concept:name', 'time:timestamp',
                                                                            'case:Rfp-id', 'Rfp-id', 'case:Cost', 'Cost'],
                                                     'categorical-attributes': ['org:role', 'org:resource', 'case:Project',
                                                                                'case:Task', 'case:OrganizationalEntity', 
                                                                                'case:Activity', 'case:RfpNumber']},
                      'Road_Traffic_Fine_Management_Process': {'not-data-attributes': ['case:concept:name', 'time:timestamp',
                                                                                      'concept:name', 'lifecycle:transition',
                                                                                      'matricola'],
                                                               'categorical-attributes': ['org:resource', 'dismissal', 'vehicleClass',
                                                                                          'article', 'points', 'notificationType', 'lastSent']}
     }
    return log_properties

class LogProperties:
    def __init__(self):
        self.properties = import_log_properties()

    def get_log_properties(self, log_name):
        return self.properties[log_name.split(".xes")[0]]
