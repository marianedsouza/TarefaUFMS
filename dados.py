"""
Conjunto de dados de texto para a atividade.

Sao 80 frases curtas distribuidas em 4 temas bem distintos:
    - Tecnologia / computacao
    - Esportes
    - Culinaria / comida
    - Natureza / meio ambiente

O rotulo real (categoria) NAO e usado pelo K-Means: ele serve apenas
para, no final, avaliarmos se os grupos descobertos de forma nao
supervisionada correspondem aos temas reais.
"""

FRASES = [
    # ---------------- Tecnologia / computacao ----------------
    ("O novo processador tem oito nucleos e alta velocidade de clock.", "tecnologia"),
    ("A inteligencia artificial esta transformando o mercado de software.", "tecnologia"),
    ("Precisamos atualizar o servidor para melhorar o desempenho do site.", "tecnologia"),
    ("A linguagem Python e muito usada em ciencia de dados.", "tecnologia"),
    ("O smartphone novo vem com uma bateria de longa duracao.", "tecnologia"),
    ("Redes neurais aprendem padroes a partir de grandes volumes de dados.", "tecnologia"),
    ("A nuvem permite armazenar arquivos e acessa-los de qualquer lugar.", "tecnologia"),
    ("Programadores usam controle de versao para colaborar em projetos.", "tecnologia"),
    ("O algoritmo de busca retorna resultados em milissegundos.", "tecnologia"),
    ("A criptografia protege os dados durante a transmissao pela internet.", "tecnologia"),
    ("O aplicativo trava quando abrimos muitas abas ao mesmo tempo.", "tecnologia"),
    ("Placas de video modernas aceleram o treino de modelos de aprendizado.", "tecnologia"),
    ("A rede sem fio caiu e perdemos a conexao com o banco de dados.", "tecnologia"),
    ("Compiladores traduzem codigo fonte em instrucoes de maquina.", "tecnologia"),
    ("O teclado mecanico tem uma resposta tatil muito agradavel.", "tecnologia"),
    ("Contêineres facilitam a implantacao de aplicacoes em qualquer ambiente.", "tecnologia"),
    ("O modelo de linguagem gera texto coerente a partir de um comando.", "tecnologia"),
    ("A memoria RAM insuficiente deixa o computador lento.", "tecnologia"),
    ("Testes automatizados ajudam a evitar erros no sistema.", "tecnologia"),
    ("O robo aspirador mapeia a casa com sensores a laser.", "tecnologia"),

    # ---------------- Esportes ----------------
    ("O jogador marcou um gol incrivel nos ultimos minutos da partida.", "esportes"),
    ("A selecao venceu o campeonato depois de uma disputa de penaltis.", "esportes"),
    ("O maratonista treinou meses para completar a corrida de rua.", "esportes"),
    ("O time de basquete converteu a cesta de tres pontos no fim.", "esportes"),
    ("A tenista sacou forte e conquistou o ponto decisivo.", "esportes"),
    ("O tecnico mudou a formacao tatica no segundo tempo.", "esportes"),
    ("Os nadadores competiram nos cem metros livres.", "esportes"),
    ("O ciclista liderou a etapa nas montanhas com folga.", "esportes"),
    ("A torcida vibrou com a vitoria no ultimo lance do jogo.", "esportes"),
    ("O goleiro defendeu o penalti e garantiu o titulo.", "esportes"),
    ("O boxeador venceu a luta por nocaute no terceiro round.", "esportes"),
    ("A ginasta executou um salto perfeito e recebeu nota maxima.", "esportes"),
    ("O piloto largou na pole position e venceu a corrida.", "esportes"),
    ("O volei brasileiro conquistou a medalha de ouro.", "esportes"),
    ("O atleta bateu o recorde mundial de salto em distancia.", "esportes"),
    ("A partida de futebol terminou empatada em dois a dois.", "esportes"),
    ("O time de rugby avancou com forca ate a linha de fundo.", "esportes"),
    ("O surfista dropou uma onda gigante no campeonato.", "esportes"),
    ("Os jogadores comemoraram a classificacao para a final.", "esportes"),
    ("O arbitro apitou o fim do jogo e a torcida invadiu o campo.", "esportes"),

    # ---------------- Culinaria / comida ----------------
    ("A receita leva farinha, ovos, acucar e uma pitada de sal.", "culinaria"),
    ("O molho de tomate fica mais saboroso com manjericao fresco.", "culinaria"),
    ("Assamos o bolo de chocolate por quarenta minutos no forno.", "culinaria"),
    ("O restaurante serve uma feijoada completa aos sabados.", "culinaria"),
    ("A massa da pizza precisa descansar antes de ir ao forno.", "culinaria"),
    ("Refogamos a cebola e o alho antes de adicionar o arroz.", "culinaria"),
    ("O cafe da manha inclui pao quentinho, manteiga e suco de laranja.", "culinaria"),
    ("A sobremesa era um pudim de leite condensado bem cremoso.", "culinaria"),
    ("Temperamos a carne com sal grosso e alecrim antes de grelhar.", "culinaria"),
    ("A sopa de legumes aquece bem nos dias frios de inverno.", "culinaria"),
    ("O chef finalizou o prato com uma reducao de vinho tinto.", "culinaria"),
    ("Batemos as claras em neve para deixar o bolo fofo.", "culinaria"),
    ("O peixe grelhado veio acompanhado de arroz e salada verde.", "culinaria"),
    ("A padaria vende croissants amanteigados logo cedo.", "culinaria"),
    ("Colocamos queijo derretido por cima do macarrao gratinado.", "culinaria"),
    ("O suco de frutas frescas e uma otima opcao no verao.", "culinaria"),
    ("A lasanha ficou pronta com camadas de molho e presunto.", "culinaria"),
    ("Deixamos a massa do pao crescer por duas horas.", "culinaria"),
    ("O hamburguer artesanal vinha com bacon e cebola caramelizada.", "culinaria"),
    ("O doce de leite caseiro acompanha bem o queijo mineiro.", "culinaria"),

    # ---------------- Natureza / meio ambiente ----------------
    ("A floresta amazonica abriga uma enorme diversidade de especies.", "natureza"),
    ("O rio corre entre as montanhas ate desaguar no mar.", "natureza"),
    ("As abelhas sao essenciais para a polinizacao das plantas.", "natureza"),
    ("O desmatamento ameaca o habitat de muitos animais silvestres.", "natureza"),
    ("A chuva forte encheu os lagos e regou as plantacoes.", "natureza"),
    ("O passaro construiu o ninho no alto da arvore.", "natureza"),
    ("As tartarugas marinhas retornam a praia para desovar.", "natureza"),
    ("O vento espalha as sementes pelo campo aberto.", "natureza"),
    ("A reciclagem ajuda a reduzir a poluicao do meio ambiente.", "natureza"),
    ("O sol nasce atras das colinas iluminando o vale.", "natureza"),
    ("A geleira derrete mais rapido por causa do aquecimento global.", "natureza"),
    ("As raizes das arvores seguram o solo e evitam a erosao.", "natureza"),
    ("O recife de corais abriga peixes de cores vibrantes.", "natureza"),
    ("A trilha na mata revela cachoeiras escondidas.", "natureza"),
    ("Os lobos vivem em matilhas nas regioes frias do norte.", "natureza"),
    ("A borboleta pousou suavemente sobre a flor amarela.", "natureza"),
    ("O deserto tem noites frias e dias muito quentes.", "natureza"),
    ("A preservacao das nascentes garante agua limpa para todos.", "natureza"),
    ("O urso hiberna durante o inverno rigoroso.", "natureza"),
    ("As folhas caem no outono e cobrem o chao da floresta.", "natureza"),
]


def carregar_dados():
    """Retorna (lista_de_textos, lista_de_rotulos_reais)."""
    textos = [t for t, _ in FRASES]
    rotulos = [r for _, r in FRASES]
    return textos, rotulos


if __name__ == "__main__":
    textos, rotulos = carregar_dados()
    print(f"Total de frases: {len(textos)}")
    from collections import Counter
    print("Distribuicao por categoria real:", dict(Counter(rotulos)))
