# -*- coding: utf-8 -*-
"""
Cria a base reprodutivel de 150 textos em portugues:
    dataset_150_textos_portugues.csv  (colunas: texto, categoria)

Sao 6 categorias tematicas x 25 frases = 150 textos. As frases foram
escritas para representar contextos semanticos bem distintos, permitindo
avaliar se o K-Means (nao supervisionado) recupera esses temas.

Rode uma vez:
    python criar_dataset.py
"""

import csv

CATEGORIAS = {
    "tecnologia": [
        "O novo processador tem oito nucleos e alta velocidade de clock.",
        "A inteligencia artificial esta transformando o mercado de software.",
        "Precisamos atualizar o servidor para melhorar o desempenho do site.",
        "A linguagem Python e muito usada em ciencia de dados.",
        "O smartphone novo vem com uma bateria de longa duracao.",
        "Redes neurais aprendem padroes a partir de grandes volumes de dados.",
        "A computacao em nuvem permite acessar arquivos de qualquer lugar.",
        "Programadores usam controle de versao para colaborar em projetos.",
        "O algoritmo de busca retorna resultados em milissegundos.",
        "A criptografia protege os dados durante a transmissao pela internet.",
        "O aplicativo trava quando abrimos muitas abas ao mesmo tempo.",
        "Placas de video modernas aceleram o treino de modelos de aprendizado.",
        "O banco de dados armazena milhoes de registros de usuarios.",
        "Compiladores traduzem codigo fonte em instrucoes de maquina.",
        "A realidade virtual cria ambientes imersivos em tres dimensoes.",
        "Conteineres facilitam a implantacao de aplicacoes em qualquer ambiente.",
        "O modelo de linguagem gera texto coerente a partir de um comando.",
        "A memoria insuficiente deixa o computador lento durante o uso.",
        "Testes automatizados ajudam a evitar erros no sistema.",
        "O robo aspirador mapeia a casa com sensores a laser.",
        "A rede de fibra optica aumentou muito a velocidade da internet.",
        "Desenvolvedores publicaram uma nova versao do aplicativo hoje.",
        "O chip de seguranca protege as senhas armazenadas no aparelho.",
        "A automacao industrial usa sensores conectados a um sistema central.",
        "O navegador bloqueia rastreadores para proteger a privacidade.",
    ],
    "esportes": [
        "O jogador marcou um gol incrivel nos ultimos minutos da partida.",
        "A selecao venceu o campeonato depois de uma disputa de penaltis.",
        "O maratonista treinou meses para completar a corrida de rua.",
        "O time de basquete converteu a cesta de tres pontos no fim.",
        "A tenista sacou forte e conquistou o ponto decisivo.",
        "O tecnico mudou a formacao tatica no segundo tempo.",
        "Os nadadores competiram na prova dos cem metros livres.",
        "O ciclista liderou a etapa nas montanhas com folga.",
        "A torcida vibrou com a vitoria no ultimo lance do jogo.",
        "O goleiro defendeu o penalti e garantiu o titulo.",
        "O boxeador venceu a luta por nocaute no terceiro round.",
        "A ginasta executou um salto perfeito e recebeu nota maxima.",
        "O piloto largou na pole position e venceu a corrida.",
        "A equipe de volei conquistou a medalha de ouro.",
        "O atleta bateu o recorde mundial de salto em distancia.",
        "A partida de futebol terminou empatada em dois a dois.",
        "O time avancou com forca ate a linha de fundo do campo.",
        "O surfista dropou uma onda gigante no campeonato.",
        "Os jogadores comemoraram a classificacao para a final.",
        "O arbitro apitou o fim do jogo e encerrou a disputa.",
        "O corredor cruzou a linha de chegada em primeiro lugar.",
        "A defesa do time foi decisiva para segurar o resultado.",
        "O treino de forca melhora o desempenho dos atletas.",
        "A delegacao conquistou varias medalhas nos jogos.",
        "O jogador foi expulso apos receber o segundo cartao amarelo.",
    ],
    "culinaria": [
        "A receita leva farinha, ovos, acucar e uma pitada de sal.",
        "O molho de tomate fica mais saboroso com manjericao fresco.",
        "Assamos o bolo de chocolate por quarenta minutos no forno.",
        "O restaurante serve uma feijoada completa aos sabados.",
        "A massa da pizza precisa descansar antes de ir ao forno.",
        "Refogamos a cebola e o alho antes de adicionar o arroz.",
        "O cafe da manha inclui pao quentinho, manteiga e suco de laranja.",
        "A sobremesa era um pudim de leite bem cremoso.",
        "Temperamos a carne com sal grosso e alecrim antes de grelhar.",
        "A sopa de legumes aquece bem nos dias frios de inverno.",
        "O chef finalizou o prato com uma reducao de vinho tinto.",
        "Batemos as claras em neve para deixar o bolo fofo.",
        "O peixe grelhado veio acompanhado de arroz e salada verde.",
        "A padaria vende croissants amanteigados logo cedo.",
        "Colocamos queijo derretido por cima do macarrao gratinado.",
        "O suco de frutas frescas e uma otima opcao no verao.",
        "A lasanha ficou pronta com camadas de molho e presunto.",
        "Deixamos a massa do pao crescer por duas horas.",
        "O hamburguer artesanal vinha com bacon e cebola caramelizada.",
        "O doce de leite caseiro acompanha bem o queijo.",
        "Servimos o risoto de cogumelos ainda bem cremoso.",
        "A salada tinha folhas verdes, tomate e azeite de oliva.",
        "O churrasco de domingo reuniu a familia no quintal.",
        "Preparamos brigadeiros para a festa de aniversario.",
        "O tempero da comida ficou perfeito com pimenta e limao.",
    ],
    "natureza": [
        "A floresta amazonica abriga uma enorme diversidade de especies.",
        "O rio corre entre as montanhas ate desaguar no mar.",
        "As abelhas sao essenciais para a polinizacao das plantas.",
        "O desmatamento ameaca o habitat de muitos animais silvestres.",
        "A chuva forte encheu os lagos e regou as plantacoes.",
        "O passaro construiu o ninho no alto da arvore.",
        "As tartarugas marinhas retornam a praia para desovar.",
        "O vento espalha as sementes pelo campo aberto.",
        "A reciclagem ajuda a reduzir a poluicao do meio ambiente.",
        "O sol nasce atras das colinas iluminando o vale.",
        "A geleira derrete mais rapido por causa do aquecimento global.",
        "As raizes das arvores seguram o solo e evitam a erosao.",
        "O recife de corais abriga peixes de cores vibrantes.",
        "A trilha na mata revela cachoeiras escondidas.",
        "Os lobos vivem em matilhas nas regioes frias do norte.",
        "A borboleta pousou suavemente sobre a flor amarela.",
        "O deserto tem noites frias e dias muito quentes.",
        "A preservacao das nascentes garante agua limpa para todos.",
        "O urso hiberna durante o inverno rigoroso.",
        "As folhas caem no outono e cobrem o chao da floresta.",
        "A savana africana e o lar de leoes, zebras e girafas.",
        "O oceano cobre a maior parte da superficie do planeta.",
        "A montanha nevada atrai alpinistas do mundo inteiro.",
        "O ecossistema depende do equilibrio entre as especies.",
        "A neblina cobriu o vale logo ao amanhecer.",
    ],
    "saude": [
        "A vacina previne diversas doencas graves na populacao.",
        "O medico receitou um antibiotico para tratar a infeccao.",
        "A pressao arterial deve ser monitorada com regularidade.",
        "Exercicios fisicos melhoram a saude do coracao.",
        "O paciente se recuperou bem apos a cirurgia.",
        "Uma dieta equilibrada previne muitas doencas cronicas.",
        "Dormir bem e fundamental para a recuperacao do corpo.",
        "O enfermeiro mediu a temperatura e a saturacao do paciente.",
        "Beber agua com frequencia ajuda a manter o corpo hidratado.",
        "A fisioterapia ajudou o joelho a recuperar o movimento.",
        "O exame de sangue detectou uma leve anemia.",
        "A consulta com o dentista preveniu problemas nos dentes.",
        "O uso de mascara reduz a transmissao de virus respiratorios.",
        "A meditacao ajuda a diminuir o estresse do dia a dia.",
        "O cardiologista recomendou reduzir o sal na alimentacao.",
        "A vacinacao infantil protege contra sarampo e poliomielite.",
        "Caminhar todos os dias melhora a circulacao sanguinea.",
        "O tratamento incluiu repouso e muita hidratacao.",
        "A nutricionista montou um cardapio rico em fibras.",
        "O hospital ampliou o numero de leitos na emergencia.",
        "A saude mental merece a mesma atencao que a fisica.",
        "O medico pediu uma tomografia para investigar a dor.",
        "Lavar as maos com frequencia previne muitas infeccoes.",
        "A pratica de alongamento evita lesoes musculares.",
        "O check-up anual ajuda a detectar doencas no inicio.",
    ],
    "educacao": [
        "A professora explicou o teorema de Pitagoras no quadro.",
        "Os alunos entregaram o trabalho de historia na data marcada.",
        "A biblioteca da escola recebeu novos livros de literatura.",
        "O estudante revisou a materia antes da prova de matematica.",
        "A universidade abriu inscricoes para os cursos de graduacao.",
        "O professor corrigiu as redacoes durante o fim de semana.",
        "A aula de geografia abordou os climas do Brasil.",
        "Os estudantes participaram de uma feira de ciencias.",
        "A escola adotou tablets para as atividades em sala.",
        "O curso online oferece certificado ao final das aulas.",
        "A monitoria ajuda os alunos com dificuldades em fisica.",
        "O vestibular avalia conhecimentos de varias disciplinas.",
        "A turma leu um classico da literatura brasileira.",
        "O laboratorio de quimica tem equipamentos para experimentos.",
        "A palestra motivou os jovens a seguir a carreira academica.",
        "O diretor anunciou o calendario do proximo semestre letivo.",
        "Os alunos apresentaram seminarios sobre o meio ambiente.",
        "A bolsa de estudos cobre a mensalidade da faculdade.",
        "A prova final avaliou todo o conteudo do ano.",
        "O intercambio permite estudar em outro pais por um ano.",
        "A pesquisa cientifica exige metodo e dedicacao.",
        "O caderno estava cheio de anotacoes sobre a aula.",
        "A educacao infantil estimula a curiosidade das criancas.",
        "O tutor explicou o exercicio passo a passo.",
        "A formatura reuniu os alunos e suas familias no auditorio.",
    ],
}


def main():
    linhas = []
    for categoria, frases in CATEGORIAS.items():
        assert len(frases) == 25, f"{categoria} tem {len(frases)} frases (esperado 25)"
        for frase in frases:
            linhas.append({"texto": frase, "categoria": categoria})

    caminho = "dataset_150_textos_portugues.csv"
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "categoria"])
        writer.writeheader()
        writer.writerows(linhas)

    print(f"Dataset criado: {caminho}")
    print(f"Total de textos: {len(linhas)}")
    print(f"Categorias: {list(CATEGORIAS.keys())}")


if __name__ == "__main__":
    main()
