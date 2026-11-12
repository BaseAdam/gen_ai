import { expect, it } from 'vitest'

import { formatSalary } from './format.ts'

it.each([
  [12000, 17000, '12,000–17,000 PLN'],
  [12000, null, 'from 12,000 PLN'],
  [null, 17000, 'up to 17,000 PLN'],
  [null, null, 'Salary not given'],
])('formats a salary range of %s and %s', (min, max, expected) => {
  expect(formatSalary(min, max)).toBe(expected)
})
