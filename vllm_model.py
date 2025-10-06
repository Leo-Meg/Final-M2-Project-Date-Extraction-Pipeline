# vllm_model.py
from vllm import LLM, SamplingParams,LLMEngine
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer,AutoConfig
import os
import json
import pandas as pd


# True to use modelscope , false to use huggingface
os.environ['VLLM_USE_MODELSCOPE']='True'

def clean_text(text):
    """
    remove space and not line
    """

    lines = text.splitlines()
    # 2. clean empty line and useless string
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # replace with empty string
        line = ' '.join(line.split())
        if line:  # save only line
            cleaned_lines.append(line)

    # 3. make a string with each line
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = f'"{cleaned_text}"'

    return cleaned_text

def get_completion(prompts, model, tokenizer=None, max_tokens=100, temperature=0.1, top_p=0.9,max_model_len=20000):
    stop_token_ids = [151329, 151336, 151338]

    # create sampling param : not used
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )

    # Initialize LLM

    llm = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        max_model_len=80000,  # Reduce this value
        gpu_memory_utilization=0.95,
        rope_scaling={
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    )


    # generate
    outputs = llm.generate(prompts, sampling_params)

    return outputs

if __name__ == "__main__":
    # initialize VLLM engine
    # model = "/home/sidney/models_llm/hub/Qwen/Qwen2___5-14B-Instruct"
    model = "/home/sidney/models_llm/hub/Qwen/Qwen2___5-7B-Instruct"
    tokenizer = None
    template = """
    Select and Return ONLY EXACTLY ONE the most accurate publication date from the given date list and reference text, THE DATE YOU CHOOSE SHOULD APPEAR IN THE Date List:
    
    2 examples：
    
    Example 1:
    [Date List]
    ['31 janvier 2024','23/11/2023','10 mars 2023','15 janvier 2024','31 janvier','15 janvier 2024','en février 2024 et','le 01/12/2023','le 01/12/2023','le 01/12/2023']
    [Reference Text]
    "https://www.cc-bocage-bourbonnais.com/images/Délibération_ZA_Enr/20231130_Tronget.pdfCOMMUNE DE TRONGET EXTRAIT DU REGISTRE DES DÉLIBÉRATIONS République Française L’an deux mil vingt-trois, le jeudi 30 novembre 2023 à 19h30, le Conseil Municipal, Département de l’Allier dûment convoqué, s’est réuni salle de la Mairie sise 8 passage de la mairie, en session Arrondissement de Moulins ordinaire, sous la présidence du Maire, Jean-Marc DUMONT. Date de convocation : Présents : Patrick AMATHIEU, Elena BARANSKI, Daniel CANTE, Alain DETERNES, 23/11/2023 Jean-Marc DUMONT, Audrey GERAUD, Patricia RAYNAUD, Pascal RAYNAUD, Sylvain RIBIER, Franck VALETTE Nombre de conseillers : Excusés : Laurent BRUN, Jean-Marc CARTE, Stéphane HERAULT, Annie WEGRZYN En exercice : 14 Présents : 10 Votants : 14 Le quorum étant atteint, le | Pouvoirs: Laurent BRUN à Franck VALETTE, Jean-Marc CARTE à Jean-Marc Conseil Municipal peut | DUMONT, Stéphane HERAULT à Pascal RAYNAUD, Annie WEGRZYN à Eléna valablement délibérer. BARANSKI Secrétaire de séance : Daniel CANTE Zones d’Accélération des Energies Renouvelables N°36/2023 La loi n° 2023-175 du 10 mars 2023 relative à l’accélération de la production d’énergies renouvelables, dite loi APER, vise à accélérer et simplifier les projets d’implantation de producteurs d’énergie et à répondre à l’enjeu de l’acceptabilité locale. En particulier, son article 15 permet aux communes de définir, après concertation avec leurs administrés, des zones d’accélération où elles souhaitent prioritairement voir des projets d’énergies renouvelables s’implanter. Les zones d’accélération (ZAENR) concernent ainsi l’implantation d'installations terrestres de production d’énergies renouvelables, ainsi que de leurs ouvrages connexes. Ces ZAENR peuvent concerner toutes les énergies renouvelables (ENR). Elles sont définies, pour chaque catégorie de sources et de types d’installation de production d’ENR, en tenant compte de la nécessaire diversification des ENR, des potentiels du territoire concerné et de la puissance d’ENR déjà installée. (L.141-5-3 du code de l’énergie) Ces zones d’accélération ne sont pas des zones exclusives. Des projets pourront être autorisés en dehors. Toutefois, un comité de projet sera obligatoire pour ces projets, afin de garantir la bonne inclusion de la commune d’implantation et des communes limitrophes dans la conception du projet, au plus tôt et en continu. Les porteurs de projets seront, quoi qu’il en soit, incités à se diriger vers ces ZAENR qui témoignent d’une volonté politique et d’une adhésion locale du projet ENR. Monsieur le Maire précise que : ° Pour un projet, le fait d’être situé en zone d’accélération ne garantit pas son autorisation, celui-ci devant, dans tous les cas, respecter les dispositions réglementaires applicables et en tout état de cause l’instruction des projets reste faite au cas par cas. + Les zones doivent être à faibles enjeux environnementaux, agricoles et paysagers. + L’article L.314-41. du code de l’énergie prévoit que les candidats retenus à l’issue d’une procédure de mise en concurrence ou d’appel à projets sont tenus de financer notamment des projets portés par la commune ou par l’établissement public de coopération intercommunale à fiscalité propre d’implantation de l’installation en faveur de la transition énergétique. + Les communes identifient par délibération du conseil municipal des zones qui sont soumises à concertation du public selon les modalités qu’elles déterminent librement. Compte tenu de ces éléments, Monsieur le Maire expose : Les propositions de zones d’accélération pour les énergies renouvelables se fondent sur les critères suivants : + Des délaissés d’infrastructures, + __ Des zones dégradées, + Des terres agricoles inexploitables, + La présence de projets déjà connus, Les ZAENR proposées à la concertation sont les suivantes : + __ Solaire photovoltaïque : sur l’ensemble des bâtiments communaux et domaine public + __ Solaire photovoltaïque au sol dont ombrières : sur le domaine public et biens publics + _ Éolien, méthanisation : pas détermination de zone + Réseau de chaleur, bois-énergie, géothermie : projets communaux ou publics Les modalités de concertation proposées sont les suivantes : + Mise à disposition des documents et d’un registre en mairie du 15 janvier 2024 au 31 janvier 2024. + Mise à disposition des documents et d’un formulaire sur le site internet de la Communauté de Communes du Bocage Bourbonnais du 15 janvier 2024 au 31 janvier 2024. Le conseil municipal procédera à l’élaboration d’un bilan de la concertation en février 2024 et apportera les éventuelles modifications aux propositions des zones d’accélération des énergies renouvelables. Monsieur le Maire propose donc au conseil municipal d’émettre un avis favorable à : + La proposition de ZAENR pour leur mise en concertation du public, + La proposition des modalités de concertation. Après en avoir délibéré et à l'unanimité des membres présents et représentés, le Conseil Municipal décide : + D'’identifier les zones d’accélération pour l’implantation d’installations terrestres de production d’énergies renouvelables ainsi que leurs ouvrages connexes mentionnées ci- après, ainsi que sur les cartes annexées à la présente décision, qui seront soumises à concertation du public ; + Valide les modalités de concertation ; + Charge le maire ou son représentant de transmettre à l’EPCI, les zones identifiées pour concertation du public. ONT VOTE POUR : 14 ONT VOTE CONTRE : / SE SONT ABSTENUS : / ACTE EXECUTOIRE Reçu par le représentant de l’Etat le 01/12/2023 et publié le 01/12/2023 Pour extrait conforme au registre des délibérations du conseil municipal, Fait à Tronget, le 01/12/2023 Le Maire, UNS Jean-Marc DUMONT"
    Answer:31 janvier 2024
    
    Example 2: 
    [Date List]
    ['06/01/2024','05 Janvier 2024','03-2024','10 mars 2023','10 mars 2023','le 06/01/2024','le 06/01/2024','19/12/2023','30/12/2023','le 05 Janvier 2024']
    [Reference Text]
    "https://mairiechars95.fr/wp-content/uploads/2024/03/Deliberation-3-Zone-dacceleration-des-energies-renouvelables.pdfEnvoyé en préfecture le 06/01/2024 RÉPUBLIQUE FRANÇAISE M À | FR | E D FC Reu en préfecturé1e06/0172024 VAL-D'OISE | Publié le ID : 095-219501426-20240105-032024-DE Extrait du registre des délibérations du conseil municipal de CHARS Séance du 05 Janvier 2024 03-2024 OBJET : Décision du conseil municipal sur les zones d'accélération des énergies renouvelables Présents : 15 Evelyne BOSSU Xavier BACHELET Ariane MARTIN Carole BOUILLONNEC Vincent DELCHOQUE Jean-Pierre BAZIN Sébastien RAVOISIER Sheila DEPUILLE Pierre-Antoine DHUICQ Patricia CHAILLOU-LEPAREUR Sylviane LEPAPE Gérard GENNISSON Nathalie GROM Philippe CHAUVET Nicolas BELANGÉ Absents et procurations : 4 Sandrine LHORSET excusée Nicolas PRIOUX excusé Agnès AGLAVE-LUCAS excusée Caroline BOURG Pouvoir à Sylviane LEPAPE Le Président a ouvert la séance et fait l’appel nominal, il a été procédé en conformité avec l’article L.2121-15 du code général des collectivités territoriales, à la nomination d'un secrétaire pris au sein du conseil. Monsieur Philippe CHAUVET est désigné pour remplir cette fonction. Pour rappel : La loi n° 2023-175 du 10 mars 2023 relative à l'accélération de la production d'énergies renouvelables vise à accélérer le développement des énergies renouvelables de manière à lutter contre le changement climatique et préserver la sécurité d'approvisionnement de la France en électricité. L'article 15 de la loi a introduit dans le code de l'énergie un dispositif de planification territoriale à la main des communes. D'ici la fin de l’année 2023, les communes sont invitées à identifier les zones d'accélération pour l'implantation d'installations terrestres de production d'énergie renouvelable. En application de l’article L141-5-3 du code de l'énergie, ces zones sont définies, pour chaque catégorie de sources et de types d'installation de production d'énergies renouvelables : éolien terrestre, photovoltaïque, méthanisation, hydroélectricité, géothermie, en tenant compte de la nécessaire diversification des énergies renouvelables en fonction des potentiels du territoire concerné et de la puissance des projets d'énergies renouvelables déjà installée. La zone d'accélération illustre la volonté de la commune d'orienter préférentiellement les projets vers des espaces qu'elle estime adaptés. Ces projets pourront bénéficier de mécanismes financiers incitatifs. En revanche, pour un projet, le fait d’être situé en zone d'accélération ne garantit pas la délivrance de son autorisation ou de son permis. Le projet doit dans tous les cas respecter les dispositions réglementaires applicables. Un projet peut également s'implanter en dehors des zones d'accélération. Dans ce cas, un comité de projet sera obligatoire. Ce comité inclura les différentes parties prenantes concernées par un projet d'énergie renouvelable, dont les communes limitrophes. Dans le cas où les zones d'accélération au niveau régional sont suffisantes pour atteindre les objectifs régionaux de développement des énergies renouvelables, la commune peut définir des zones d'exclusion de ces projets. Le conseil municipal de la commune de Chars, régulièrement convoqué, s'est réuni sous la présidence de Madame BOSSU Evelyne, afin de délibérer sur les zones d'accélération proposée par la commune sur son territoire. Madame le Maire constate que le conseil réunit les conditions pour délibérer valablement. Vu la loi n° 2023-175 du 10 mars 2023 relative à l'accélération de la production d'énergies renouvelables, notamment son article 15, 2, rue de Gisors 95750 Chars - Téléphone 01 30 39 72 36 - Télécopie 01 30 39 94 64 Envoyé en préfecture le 06/01/2024 Reçu en préfecture le 06/01/2024 Publié le ID : 095-219501426-20240105-032024-DE Madame le Maire présente les zones identifiées comme zones d'accélération pour le développement des énergies renouvelables ainsi que les arguments ayant conduit à ces propositions de zones. Conformément à la loi, une consultation du public a été effectuée du 19/12/2023 au 30/12/2023 selon les modalités suivantes : sur le site internet de la Commune et dossier consultable en libre d'accès en Mairie. - La Commune de Chars souhaite donc s'orienter principalement vers le développement de l’énergies solaire et a identifié, dans ce cadre, deux solutions : a/ Les ombrières photovoltaïques sur le parking de la gare, b/ Le photovoltaïque de toiture sur différents bâtiments communaux suffisamment dimensionnées pour accueillir des structures viables économiquement et sur l'ensemble des toitures de la ZA des 9 arpents. (détails des zones en annexe) Madame le Maire soumet cette proposition de zones à délibération. Suite à l'exposé de Madame le Maire et après avoir délibéré à l'unanimité des présents, le conseil municipal : - DEFINIT comme zones d'accélération des énergies renouvelables de la commune les zones proposées figurant en annexe à la présente délibération - __ VALIDE la transmission de la cartographie de ces zones à Madame le sous-préfet, référent préfectoral à l'instruction des projets d'énergies renouvelables et des projets industriels nécessaires à la transition énergétique, du département de Chars, ainsi qu’à la Communauté de Commune Vexin Centre dont elles sont membres. Certifié exécutoire À CHARS, le 05 Janvier 2024 compte tenu de la transmission Evelyne BOSSU, en sous-préfecture, le ….. et de la publication, le …"
    Answer: 06/01/2024
    
    Current Task:
    [Date List]
    {Event_List}
    [Reference Text]
    {text}
    
    Rules:
    1. Output exactly one date
    2. Must be a date from the Date List
    3. No explanations, no prefix, Only return result after assistant!
    4. Keep original date format
    
    """



    

    dataframe_path = "./dataset_valid_ner.csv"
    dataframe = pd.read_csv(dataframe_path)
    file_list = dataframe['local_filename'].to_list()
    # print(file_list[:40])
    results_all ={}
    counter = 1
    total_file = len(file_list)
    for file in file_list:
        total_file -= counter
        print(f"now start processing {file}, left{total_file} files  ")
        row = dataframe[dataframe['local_filename']==file]
        context = row['text_content'].iloc[0]# Preventing return series
        # clean " \n"
        context_cleaned= clean_text(context)
        context_cleaned = context_cleaned[:4000]

        timelist = row['time_list'].iloc[0]
        input = template.format(Event_List=timelist,text=context_cleaned)
        print(input)

        outputs = get_completion(input, model, tokenizer=tokenizer, max_tokens=25, temperature=0.08, top_p=0.95)
        for output in outputs:
            results = output.outputs[0].text
            print(results)
        results_all[file] = results
        # print(results_all)
        # break
    dataframe["predicted_time"] = dataframe['local_filename'].map(results_all)
    df = dataframe[['doc_id','url','cache','text version','nature','published','entity','entity_type','predicted_time','Gold_label']]
    df.to_csv("final_results_predicted.csv",index=False)
