import { ColumnType } from 'antd/es/table';
import React, { useMemo } from 'react';

import ResponsiveTable from 'components/ResponsiveTable';
import { isObject } from 'shared/utils/data';
import { ExperimentBase, TrialDetails } from 'types';
import { alphaNumericSorter } from 'utils/sort';

import css from './TrialDetailsHyperparameters.module.scss';

export interface Props {
  experiment: ExperimentBase;
  trial: TrialDetails;
}

interface HyperParameter {
  hyperparameter: string,
  value: string,
}

const TrialDetailsHyperparameters: React.FC<Props> = ({ trial }: Props) => {
  const columns: ColumnType<HyperParameter>[] = useMemo(() => [
    {
      dataIndex: 'hyperparameter',
      defaultSortOrder: 'ascend',
      key: 'hyperparameter',
      sorter: (a: HyperParameter, b: HyperParameter) =>
        alphaNumericSorter(a.hyperparameter, b.hyperparameter),
      title: 'Hyperparameter',
    },
    {
      dataIndex: 'value',
      key: 'value',
      title: 'Value',
    },
  ], []);

  const dataSource: HyperParameter[] = useMemo(() => {
    if (trial?.hyperparameters == null) return [];
    return Object.entries(trial.hyperparameters).map(([ hyperparameter, value ]) => {
      return {
        hyperparameter,
        key: hyperparameter,
        value: isObject(value) ? JSON.stringify(value, null, 2) : String(value),
      };
    });
  }, [ trial?.hyperparameters ]);

  return (
    <div className={css.base}>
      <ResponsiveTable
        columns={columns}
        dataSource={dataSource}
        pagination={false}
        size="small"
      />
    </div>
  );
};

export default TrialDetailsHyperparameters;
