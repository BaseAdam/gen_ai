import type { Offer } from './api.ts'
import { formatSalary } from './format.ts'

export function OfferList({ offers }: { offers: Offer[] }) {
  if (offers.length === 0) {
    return null
  }
  return (
    <section aria-labelledby="offers-heading">
      <h2 id="offers-heading">{offers.length === 1 ? 'Matching offer' : 'Matching offers'}</h2>
      <ul className="offers">
        {offers.map((offer) => (
          <li key={offer.title} className="offer">
            <div className="offer-head">
              <h3>
                <mark>{offer.title}</mark>
              </h3>
              <p className="salary">{formatSalary(offer.salary_min, offer.salary_max)}</p>
            </div>
            <p className="company">
              {offer.company}, {offer.city}
              {offer.remote && <span className="remote">Remote</span>}
            </p>
            {offer.skills.length > 0 && <p className="skills">{offer.skills.join(', ')}</p>}
          </li>
        ))}
      </ul>
    </section>
  )
}
