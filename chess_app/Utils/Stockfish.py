import chess
import chess.engine

class Stockfish:
    stockfish_path = "/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish"
    
    @staticmethod
    def get_move(fen_str, time=2):
        try:
            board = chess.Board(fen_str)
            with chess.engine.SimpleEngine.popen_uci(Stockfish.stockfish_path) as engine:
                limit = chess.engine.Limit(time=time)
                result = engine.play(board, limit)
                best_move = result.move
                return best_move
        except Exception as e:
            print(e)
            return None



