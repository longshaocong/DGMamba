import torch 

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))
        
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, 
            batch_size = batch_size, 
            drop_last = True
        )

        self.infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset, 
            num_workers= num_workers, 
            batch_sampler=InfiniteSampler(batch_sampler),
            pin_memory=True
        ))

    def __iter__(self):
        while True:
            yield next(self.infinite_iterator)

    def __len__(self):
        raise ValueError
        
class FastDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=InfiniteSampler(batch_sampler), 
            pin_memory=True
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
