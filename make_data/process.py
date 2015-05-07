

def load_data(f):
    line = f.readline()
    if line == '':
        return
    game = [];
    while len(line.split()) == 2:
        board = ''
        for i in xrange(12):
            board += f.readline().split()[0]
        game.append([line, board])
        line = f.readline()
    win = line.split()[0]
    if win == 'E' or win == 'X':
        return []
    if win == 'B':
        for board in game:
            board[1] = board[1].replace('B', 'C')
            board[1] = board[1].replace('A', 'B')
            board[1] = board[1].replace('C', 'A')

    game = filter(lambda x: x[0].split()[1] == win, game)
    game = map(lambda x: (x[0].split()[0], x[1]), game)
    return game

if __name__ == "__main__":
    f = open('result/result.txt', 'r')

    while True:
        match = load_data(f)
        if match == None:
            break
        for x in match[-10:]:
            print x[0], x[1]
