from abc import ABC, abstractmethod


class BaseConnector(ABC):

    source_id = ""
    cadence = ""

    @abstractmethod
    def pull(self, start, end):
        pass

    @abstractmethod
    def normalize(self, raw):
        pass

    def run(self, start, end):

        raw_records = self.pull(
            start,
            end
        )

        normalized = [
            self.normalize(r)
            for r in raw_records
        ]

        return normalized