import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')

# def extract_name_relations(doc):
#     # Merge entities and noun chunks into one token
#     spans = list(doc.ents) + list(doc.noun_chunks)
#     spans = spacy.util.filter_spans(spans)
#     with doc.retokenize() as retokenizer:
#         for span in spans:
#             retokenizer.merge(span)
#     relations = []
#     for person in filter(lambda w: w.ent_type_ == "PERSON", doc):
#         if person.dep_ in ("attr", "dobj"):
#             subject = [w for w in person.head.lefts if w.dep_ == "nsubj"]
#             if subject:
#                 subject = subject[0]
#                 relations.append((subject, person))
#         elif person.dep_ == "pobj" and person.head.dep_ == "prep":
#             relations.append((person.head.head, person))
#     return relations

# relations = extract_name_relations(doc)
# for r1, r2 in relations:
#     print(f'{r1.text}\t{r2.ent_type_}\t{r2.text}')

# for r1, r2 in relations:
#     if r2.dep_ == 'pobj': employee = r2.text
#     elif r2.dep_ == 'attr': name = r2.text

# print(f'name: {name}\temployee: {employee}')

# more friendly way
def get_relations(doc):
    client = None
    employees = []
    for person in filter(lambda token: token.ent_type_ == 'PERSON', doc):
        if person.dep_ == 'pobj' or person.dep_ == 'conj' and person in employees:
            employees.append(person)
        elif person.dep_ == 'attr':
            client = person
    print(f'client: {client}\temployees: {employees}')


if __name__ == '__main__':
    text1 = "I'd like to schedule an appointment with John at 4pm on Sunday, the name is Kyle."
    text2 = "I would like to schedule an appointment with John or Carlos at 4pm on Sunday for Jess."
    text3 = "I want to schedule an appointment with Carlos or Abby at 4pm on Sunday for John."
    employees = ['John', 'Abby', 'Carlos', 'Ryan']

    doc = nlp(text2)

    # get_relations(doc)
    for token in doc:
        # print(token, '-', token.conjuncts)
        print(token, '-', token.ent_type_, '-', token.head)
    