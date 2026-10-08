import { Link, NavLink } from "react-router-dom";
import { LOGO_SRC, PRODUCT_NAME, PRODUCT_SUBTITLE, useLogoAvailable } from "../brand";

export default function AppHeader() {
  const logo = useLogoAvailable();
  const cls = ({ isActive }: { isActive: boolean }) => `nav-link${isActive ? " active" : ""}`;
  return (
    <header className="app-header">
      <div className="app-header-inner">
        <Link to="/" className="brand" aria-label={`${PRODUCT_NAME}, home`}>
          {logo && (
            <>
              <img className="brand-logo" src={LOGO_SRC} alt="" />
              <span className="brand-divider" aria-hidden="true" />
            </>
          )}
          <span className="brand-text">
            <span className="brand-name">
              {/* "Swedbank" is dropped on small screens when the logo already says it */}
              <span className={logo ? "brand-prefix" : undefined}>Swedbank </span>ESG Compass
            </span>
            <span className="brand-subtitle">{PRODUCT_SUBTITLE}</span>
          </span>
        </Link>
        <nav className="main-nav" aria-label="Main">
          <NavLink to="/" end className={cls}>
            New Assessment
          </NavLink>
          <NavLink to="/assessments" className={cls}>
            Assessments
          </NavLink>
          <NavLink to="/questionnaire" className={cls}>
            Questionnaire Example
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
