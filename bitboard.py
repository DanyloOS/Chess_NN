def bitboard(board):
    w_pawn = (np.asarray(board.pieces(chess.PAWN, chess.WHITE).tolist())).astype(int)
    w_rook = (np.asarray(board.pieces(chess.ROOK, chess.WHITE).tolist())).astype(int)
    w_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.WHITE).tolist())).astype(int)
    w_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.WHITE).tolist())).astype(int)
    w_queen = (np.asarray(board.pieces(chess.QUEEN, chess.WHITE).tolist())).astype(int)
    w_king = (np.asarray(board.pieces(chess.KING, chess.WHITE).tolist())).astype(int)

    b_pawn = (np.asarray(board.pieces(chess.PAWN, chess.BLACK).tolist())).astype(int)
    b_rook = (np.asarray(board.pieces(chess.ROOK, chess.BLACK).tolist())).astype(int)
    b_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.BLACK).tolist())).astype(int)
    b_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.BLACK).tolist())).astype(int)
    b_queen = (np.asarray(board.pieces(chess.QUEEN, chess.BLACK).tolist())).astype(int)
    b_king = (np.asarray(board.pieces(chess.KING, chess.BLACK).tolist())).astype(int)
    
    testFen = board.fen(en_passant="fen")
    bit12WhoToMove = 1 if testFen.split()[1] == "b" else 0
    numericLetter = -1
    rankNumber = 0
    if '-' not in testFen.split()[3]:
        numericLetter = ord(testFen.split()[3][0]) - ord('a')
        rankNumber = int(testFen.split()[3][1]) - 1
    
    isEnPassant = 0 if testFen.split()[3][0] == '-' else 1
    castlingStr = testFen.split()[2]
    castling = ((1<<3) if "q" in castlingStr else 0) + ((1<<2) if "k" in castlingStr else 0)\
             + ((1<<1) if "Q" in castlingStr else 0) + (1 if "K" in castlingStr else 0)
    
    firstInt = int(testFen.split()[-2])
    secondInt = 0
    thirdInt = int(testFen.split()[-1])
    forthInt = (rankNumber << 3) + numericLetter
    fifthInt = (isEnPassant << 5) + (bit12WhoToMove << 4) + castling
    
    add_info = np.array([firstInt, secondInt, thirdInt, forthInt, fifthInt], dtype=np.uint8)
    add_info = np.unpackbits(add_info, axis=0).astype(np.single) 

    customBitboard = np.concatenate((w_king, w_queen, w_rook, w_bishop, w_knight, w_pawn,
                                     b_king, b_queen, b_rook, b_bishop, b_knight, b_pawn,
                                     add_info))
    
    return customBitboard
