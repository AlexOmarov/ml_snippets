from dataclasses import dataclass


@dataclass
class TheoryResult:
    data: str

    def serialize(self):
        return {
            'data': self.data
        }
